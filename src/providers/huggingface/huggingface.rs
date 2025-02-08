use super::dtype_utils::{get_dtype, validate_dtype_compatibility};
use super::tokenizer::load_tokenizer;
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::de::DeserializeOwned;
use std::env;

use crate::models::model_initializer::ModelInitializer;
use crate::models::Model;

pub async fn load_model<M: ModelInitializer>(
    model_id: &str,
    revision: &str,
    default_dtype: DType,
    device: &Device,
) -> Result<Model<M>>
where
    M::Config: DeserializeOwned,
{
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
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;
    let config_path = repo
        .get("config.json")
        .context("Failed to download config.json")?;

    tracing::info!("Loading tokenizer from {:?}", tokenizer_path);
    let tokenizer =
        load_tokenizer(model_id, &tokenizer_path).context("Failed to load tokenizer")?;

    tracing::info!("Loading model configuration");
    let config_content =
        std::fs::read_to_string(&config_path).context("Failed to read config.json")?;
    let config_file: M::Config =
        serde_json::from_str(&config_content).context("Failed to parse config.json")?;

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
        let index_path = repo
            .get("model.safetensors.index.json")
            .context("Failed to find either model.safetensors or model.safetensors.index.json")?;

        let index_content = std::fs::read_to_string(&index_path)
            .context("Failed to read model.safetensors.index.json")?;
        let index: serde_json::Value = serde_json::from_str(&index_content)
            .context("Failed to parse model.safetensors.index.json")?;

        let weight_map = index["weight_map"]
            .as_object()
            .context("Invalid index file format: missing or invalid weight_map")?;
        let model_files: std::collections::HashSet<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        tracing::info!("Downloading {} model files", model_files.len());
        let mut all_weights = Vec::new();

        for filename in model_files {
            tracing::info!("Downloading {}", filename);
            let file_path = repo
                .get(&filename)
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

    let dtype = get_dtype(None, default_dtype); // We'll get dtype from model-specific config
    validate_dtype_compatibility(dtype, model_id);

    let (model, cache) = M::initialize_model(&config_file, tensors, dtype, device)?;

    tracing::info!("Model loaded successfully");
    Ok(Model::new(tokenizer, model, device.clone(), cache))
}
