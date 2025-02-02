use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mistral::Config as MistralConfig;
use candle_transformers::models::llama::{Cache, Llama}; // Use Llama model temporarily since Mistral isn't available yet
use candle_nn::Activation;
use serde::Deserialize;
use std::collections::HashMap;

use super::model_initializer::ModelInitializer;

// Temporarily use Llama as the base model since Mistral isn't available yet
pub type Mistral = Llama;

#[derive(Deserialize, Clone)]
pub struct ConfigFile {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>, // Keep as f64 for Mistral config
    pub max_position_embeddings: Option<usize>,
    pub sliding_window: Option<usize>,
    pub torch_dtype: Option<String>,
}

impl From<ConfigFile> for MistralConfig {
    fn from(cf: ConfigFile) -> Self {
        Self {
            hidden_size: cf.hidden_size,
            intermediate_size: cf.intermediate_size,
            vocab_size: cf.vocab_size,
            num_hidden_layers: cf.num_hidden_layers,
            num_attention_heads: cf.num_attention_heads,
            num_key_value_heads: cf.num_key_value_heads.unwrap_or(cf.num_attention_heads),
            rms_norm_eps: cf.rms_norm_eps,
            rope_theta: cf.rope_theta.unwrap_or(10000.0), // This stays as f64 for Mistral
            max_position_embeddings: cf.max_position_embeddings.unwrap_or(4096),
            sliding_window: Some(cf.sliding_window.unwrap_or(4096)),
            use_flash_attn: false,
            head_dim: None, // Will be computed automatically
            hidden_act: Activation::Silu,
        }
    }
}

pub struct MistralWithConfig {
    model: Mistral,
    config: MistralConfig,
}

impl ModelInitializer for MistralWithConfig {
    type Config = ConfigFile;
    type Cache = Cache;

    fn initialize_model(
        config: &Self::Config,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<(Self, Self::Cache)> {
        let mistral_config = MistralConfig::from(config.clone());
        tracing::debug!(
            "Model config: hidden_size={}, layers={}, heads={}", 
            mistral_config.hidden_size, mistral_config.num_hidden_layers, mistral_config.num_attention_heads
        );

        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        
        tracing::info!("Initializing model cache");
        // Convert MistralConfig to LlamaConfig for cache initialization
        let llama_config = candle_transformers::models::llama::Config {
            hidden_size: mistral_config.hidden_size,
            intermediate_size: mistral_config.intermediate_size,
            vocab_size: mistral_config.vocab_size,
            num_hidden_layers: mistral_config.num_hidden_layers,
            num_attention_heads: mistral_config.num_attention_heads,
            num_key_value_heads: mistral_config.num_key_value_heads,
            rms_norm_eps: mistral_config.rms_norm_eps,
            rope_theta: mistral_config.rope_theta as f32, // Convert to f32 for Llama config
            max_position_embeddings: mistral_config.max_position_embeddings,
            use_flash_attn: mistral_config.use_flash_attn,
            rope_scaling: None,
            tie_word_embeddings: false,
            eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(2)),
            bos_token_id: Some(1),
        };
        
        let cache = Cache::new(true, dtype, &llama_config, device)?;
        
        tracing::info!("Initializing model");
        // Temporarily use Llama model since Mistral isn't available yet
        let model = Llama::load(vb, &llama_config)?;

        Ok((Self { model, config: mistral_config }, cache))
    }

    fn initialize_cache(device: &Device, dtype: DType) -> Result<Self::Cache> {
        // Default config values for Mistral-7B
        let default_config = candle_transformers::models::llama::Config {
            hidden_size: 4096,
            intermediate_size: 14336,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-5,
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