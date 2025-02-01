use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config as LlamaConfig, Cache};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use super::Model;

// Define a custom config that we can deserialize from JSON
#[derive(serde::Deserialize)]
struct ConfigFile {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    rms_norm_eps: f64,
    rope_theta: Option<f32>,
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
        }
    }
}

pub async fn load_model(
    model_id: &str,
    revision: &str,
    dtype: DType,
    device: &Device,
) -> Result<Model> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    // Download the tokenizer and model files
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_path = repo.get("model.safetensors")?;
    let config_path = repo.get("config.json")?;

    // Load the tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Load and parse the config file
    let config_content = std::fs::read_to_string(config_path)?;
    let config_file: ConfigFile = serde_json::from_str(&config_content)?;
    let config = LlamaConfig::from(config_file);

    // Load the model weights
    let tensors = candle_core::safetensors::load(model_path, device)?;
    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    
    // Initialize cache for the model
    let cache = Cache::new(true, dtype, &config, device)?;
    
    // Initialize the model
    let model = candle_transformers::models::llama::Llama::load(vb, &cache, &config)?;

    Ok(Model::new(tokenizer, model, device.clone()))
}

pub fn get_model_files(model_id: &str, revision: &str) -> Result<Vec<PathBuf>> {
    let api = Api::new()?;
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
