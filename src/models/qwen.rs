use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, Activation};
use candle_transformers::models::qwen2::Config as QwenConfig;
use candle_transformers::models::llama::Cache; // Qwen uses Llama's cache implementation
use serde::Deserialize;
use std::collections::HashMap;

use super::model_initializer::ModelInitializer;

// Temporarily use Llama as the base model since Qwen isn't fully available yet
pub type Qwen = candle_transformers::models::llama::Llama;

#[derive(Deserialize, Clone)]
pub struct ConfigFile {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>,
    pub max_position_embeddings: Option<usize>,
    pub sliding_window: Option<usize>,
    pub torch_dtype: Option<String>,
}

impl From<ConfigFile> for QwenConfig {
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
            max_position_embeddings: cf.max_position_embeddings.unwrap_or(2048),
            sliding_window: cf.sliding_window.unwrap_or(4096),
            max_window_layers: 1, // Default value
            tie_word_embeddings: false,
            use_sliding_window: cf.sliding_window.is_some(),
            hidden_act: Activation::Silu,
        }
    }
}

pub struct QwenWithConfig {
    model: Qwen,
    config: QwenConfig,
}

impl ModelInitializer for QwenWithConfig {
    type Config = ConfigFile;
    type Cache = Cache;

    fn initialize_model(
        config: &Self::Config,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<(Self, Self::Cache)> {
        let qwen_config = QwenConfig::from(config.clone());
        tracing::debug!(
            "Model config: hidden_size={}, layers={}, heads={}",
            qwen_config.hidden_size, qwen_config.num_hidden_layers, qwen_config.num_attention_heads
        );

        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        tracing::info!("Initializing model cache");
        // Convert QwenConfig to LlamaConfig for cache initialization
        let llama_config = candle_transformers::models::llama::Config {
            hidden_size: qwen_config.hidden_size,
            intermediate_size: qwen_config.intermediate_size,
            vocab_size: qwen_config.vocab_size,
            num_hidden_layers: qwen_config.num_hidden_layers,
            num_attention_heads: qwen_config.num_attention_heads,
            num_key_value_heads: qwen_config.num_key_value_heads,
            rms_norm_eps: qwen_config.rms_norm_eps,
            rope_theta: qwen_config.rope_theta as f32,
            max_position_embeddings: qwen_config.max_position_embeddings,
            use_flash_attn: false,
            rope_scaling: None,
            tie_word_embeddings: qwen_config.tie_word_embeddings,
            eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(2)),
            bos_token_id: Some(1),
        };

        let cache = Cache::new(true, dtype, &llama_config, device)?;

        tracing::info!("Initializing model");
        // Temporarily use Llama model since Qwen isn't fully available yet
        let model = Qwen::load(vb, &llama_config)?;

        Ok((Self { model, config: qwen_config }, cache))
    }

    fn initialize_cache(device: &Device, dtype: DType) -> Result<Self::Cache> {
        // Default config values for Qwen-7B
        let default_config = candle_transformers::models::llama::Config {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 151936,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 32768,
            use_flash_attn: false,
            rope_scaling: None,
            tie_word_embeddings: false,
            eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(2)),
            bos_token_id: Some(1),
        };

        Cache::new(true, dtype, &default_config, device)
            .map_err(|e| anyhow::anyhow!("Failed to initialize cache: {}", e))
    }

    fn forward(
        &self,
        input: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        Ok(self.model.forward(input, pos, cache)?)
    }
}