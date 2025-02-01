use anyhow::{Result, Context};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config as LlamaConfig, Cache, LlamaEosToks};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use super::Model;

// Define a custom config that we can deserialize from JSON
#[derive(serde::Deserialize, Clone)]
struct ConfigFile {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    rms_norm_eps: f64,
    rope_theta: Option<f32>,
    max_position_embeddings: Option<usize>,
    torch_dtype: Option<String>,
}

fn torch_dtype_to_candle(dtype: &str) -> DType {
    match dtype {
        "float32" | "float64" => DType::F32,
        "float16" | "bfloat16" => DType::BF16, // Map both float16 and bfloat16 to BF16
        _ => DType::F32, // default to F32 for unknown types
    }
}

impl From<ConfigFile> for LlamaConfig {
    fn from(cf: ConfigFile) -> Self {
        Self {
            hidden_size: cf.hidden_size,
            intermediate_size: cf.intermediate_size,
            vocab_size: cf.vocab_size,
            num_hidden_layers: cf.num_hidden_layers,
            num_attention_heads: cf.num_attention_heads,
            num_key_value_heads: cf.num_key_value_heads.unwrap_or(cf.num_attention_heads),
            rms_norm_eps: cf.rms_norm_eps,
            rope_theta: cf.rope_theta.unwrap_or(10000.0),
            use_flash_attn: false,
            eos_token_id: Some(LlamaEosToks::Single(2)),  // Common EOS token ID for LLaMA models
            bos_token_id: Some(1),
            rope_scaling: None,
            tie_word_embeddings: false,
            max_position_embeddings: cf.max_position_embeddings.unwrap_or(4096),
        }
    }
}

pub async fn load_model(
    model_id: &str,
    revision: &str,
    default_dtype: DType,
    device: &Device,
) -> Result<Model> {
    tracing::info!("Initializing HuggingFace API client");
    let api = Api::new()
        .context("Failed to initialize HuggingFace API client")?;
    
    tracing::info!("Creating repository reference for {}", model_id);
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    tracing::info!("Downloading model files");
    // Download the tokenizer and model files
    let tokenizer_path = repo.get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;
    let model_path = repo.get("model.safetensors")
        .context("Failed to download model.safetensors")?;
    let config_path = repo.get("config.json")
        .context("Failed to download config.json")?;

    tracing::info!("Loading tokenizer from {:?}", tokenizer_path);
    // Load the tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    tracing::info!("Loading model configuration");
    // Load and parse the config file
    let config_content = std::fs::read_to_string(&config_path)
        .context("Failed to read config.json")?;
    let config_file: ConfigFile = serde_json::from_str(&config_content)
        .context("Failed to parse config.json")?;
    let config_file_clone = config_file.clone();
    let config = LlamaConfig::from(config_file);
    tracing::debug!("Model config: hidden_size={}, layers={}, heads={}", 
        config.hidden_size, config.num_hidden_layers, config.num_attention_heads);

    tracing::info!("Loading model weights");
    // Load the model weights
    let weights = std::fs::read(&model_path)?;
    let tensors = candle_core::safetensors::load_buffer(&weights, device)
        .context("Failed to load tensors from safetensors")?;
    // Use the model's dtype if available, otherwise fall back to the default
    let dtype = config_file_clone.torch_dtype
        .as_ref()
        .map(|dt| {
            tracing::info!("Model config specifies torch_dtype: {}", dt);
            let candle_dtype = torch_dtype_to_candle(dt);
            tracing::info!("Converted to candle dtype: {:?}", candle_dtype);
            candle_dtype
        })
        .unwrap_or_else(|| {
            tracing::info!("No torch_dtype specified, using default: {:?}", default_dtype);
            default_dtype
        });

    tracing::info!("Final dtype being used: {:?}", dtype);
    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    
    tracing::info!("Initializing model cache");
    // Initialize cache for the model
    let cache = Cache::new(true, dtype, &config, device)
        .context("Failed to initialize model cache")?;
    
    tracing::info!("Initializing model");
    // Initialize the model
    let model = candle_transformers::models::llama::Llama::load(vb, &config)
        .context("Failed to initialize model")?;

    tracing::info!("Model loaded successfully");
    Ok(Model::new(tokenizer, model, device.clone(), cache))
}

pub fn get_model_files(model_id: &str, revision: &str) -> Result<Vec<PathBuf>> {
    let api = Api::new()
        .context("Failed to initialize HuggingFace API client")?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let files = vec![
        repo.get("tokenizer.json")?,
        repo.get("model.safetensors")?,
        repo.get("config.json")?,
    ];

    Ok(files)
}
