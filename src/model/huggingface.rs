use std::env;
use anyhow::{Result, Context};
use candle_core::{DType, Device};
use hf_hub::{api::sync::Api, api::sync::ApiBuilder, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use super::Model;
use super::llama::{self, ConfigFile};

pub async fn load_model(
    model_id: &str,
    revision: &str,
    default_dtype: DType,
    device: &Device,
) -> Result<Model> {
    tracing::info!("Initializing HuggingFace API client");

    let token = env::var("HF_TOKEN").ok();
    let api = ApiBuilder::new()
        .with_token(token)
        .build()
        .context("Failed to initialize HuggingFace API client")?;

    tracing::info!("Creating repository reference for {}", model_id);
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    tracing::info!("Downloading model files");
    // Download the tokenizer and config files
    let tokenizer_path = repo.get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;
    let config_path = repo.get("config.json")
        .context("Failed to download config.json")?;

    tracing::info!("Loading tokenizer from {:?}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    tracing::info!("Loading model configuration");
    let config_content = std::fs::read_to_string(&config_path)
        .context("Failed to read config.json")?;
    let config_file: ConfigFile = serde_json::from_str(&config_content)
        .context("Failed to parse config.json")?;

    tracing::info!("Loading model weights");
    // First try the single file approach
    let model_path = repo.get("model.safetensors");

    let tensors = if let Ok(model_path) = model_path {
        tracing::info!("Loading single model file");
        let weights = std::fs::read(&model_path)?;
        candle_core::safetensors::load_buffer(&weights, device)
            .context("Failed to load tensors from safetensors")?
    } else {
        // If single file not found, try the index approach
        tracing::info!("Single model file not found, trying split files");
        let index_path = repo.get("model.safetensors.index.json")
            .context("Failed to find either model.safetensors or model.safetensors.index.json")?;

        let index_content = std::fs::read_to_string(&index_path)
            .context("Failed to read model.safetensors.index.json")?;
        let index: serde_json::Value = serde_json::from_str(&index_content)
            .context("Failed to parse model.safetensors.index.json")?;

        let weight_map = index["weight_map"].as_object()
            .context("Invalid index file format: missing or invalid weight_map")?;
        let model_files: std::collections::HashSet<String> = weight_map.values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        tracing::info!("Downloading {} model files", model_files.len());
        let mut all_weights = Vec::new();

        for filename in model_files {
            tracing::info!("Downloading {}", filename);
            let file_path = repo.get(&filename)
                .with_context(|| format!("Failed to download {}", filename))?;
            let weights = std::fs::read(&file_path)
                .with_context(|| format!("Failed to read {}", filename))?;
            all_weights.push(weights);
        }

        let mut combined_tensors = std::collections::HashMap::new();
        for weights in all_weights {
            let tensors = candle_core::safetensors::load_buffer(&weights, device)
                .context("Failed to load tensors from safetensors")?;
            combined_tensors.extend(tensors);
        }
        combined_tensors
    };

    let dtype = config_file.torch_dtype
        .as_ref()
        .map(|dt| {
            tracing::info!("Model config specifies torch_dtype: {}", dt);
            let candle_dtype = llama::torch_dtype_to_candle(dt);
            tracing::info!("Converted to candle dtype: {:?}", candle_dtype);
            candle_dtype
        })
        .unwrap_or_else(|| {
            tracing::info!("No torch_dtype specified, using default: {:?}", default_dtype);
            default_dtype
        });

    tracing::info!("Final dtype being used: {:?}", dtype);

    let (model, cache) = llama::initialize_model(&config_file, tensors, dtype, device)?;

    tracing::info!("Model loaded successfully");
    Ok(Model::new(tokenizer, model, device.clone(), cache))
}
