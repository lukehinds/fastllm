use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config as LlamaConfig, Cache, LlamaEosToks, Llama as CandelLlama};

pub type Llama = CandelLlama;
use serde::Deserialize;
use std::collections::HashMap;

use super::model_initializer::ModelInitializer;

// Define a custom config that we can deserialize from JSON
#[derive(Deserialize, Clone)]
pub struct ConfigFile {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f32>,
    pub max_position_embeddings: Option<usize>,
    pub torch_dtype: Option<String>,
}

// pub fn torch_dtype_to_candle(dtype: &str) -> DType {
//     match dtype {
//         "float32" | "float64" => DType::F32,
//         "float16" | "bfloat16" => DType::BF16, // Map both float16 and bfloat16 to BF16
//         _ => DType::F32, // default to F32 for unknown types
//     }
// }

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

pub struct LlamaWithConfig {
    model: Llama,
    config: LlamaConfig,
}

impl ModelInitializer for LlamaWithConfig {
    type Config = ConfigFile;
    type Cache = Cache;

    fn initialize_model(
        config: &Self::Config,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<(Self, Self::Cache)> {
        let llama_config = LlamaConfig::from(config.clone());
        tracing::debug!(
            "Model config: hidden_size={}, layers={}, heads={}", 
            llama_config.hidden_size, llama_config.num_hidden_layers, llama_config.num_attention_heads
        );

        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        
        tracing::info!("Initializing model cache");
        let cache = Cache::new(true, dtype, &llama_config, device)
            .context("Failed to initialize model cache")?;
        
        tracing::info!("Initializing model");
        let model = Llama::load(vb, &llama_config)
            .context("Failed to initialize model")?;

        Ok((Self { model, config: llama_config }, cache))
    }

    fn initialize_cache(device: &Device, dtype: DType) -> Result<Self::Cache> {
        // This method can't access `self`, so we need to create a default config
        // These values are for TinyLlama-1.1B-Chat-v1.0
        let default_config = LlamaConfig {
            hidden_size: 2048,
            intermediate_size: 5632,
            vocab_size: 32000,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            use_flash_attn: false,
            eos_token_id: Some(LlamaEosToks::Single(2)),
            bos_token_id: Some(1),
            rope_scaling: None,
            tie_word_embeddings: false,
            max_position_embeddings: 2048,
        };
        Cache::new(true, dtype, &default_config, device)
            .context("Failed to initialize model cache")
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
