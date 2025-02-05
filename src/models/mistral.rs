use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mistral::{Config as MistralConfig, Model as Mistral};
use candle_nn::Activation;
use serde::Deserialize;
use std::collections::HashMap;
use std::cell::RefCell;

// use super::token_output_stream::TokenOutputStream;
use super::model_initializer::ModelInitializer;


// Wrapper type for cache management
#[derive(Debug)]
pub struct MistralCache {
    seqlen_offset: usize,
}

// Implement Send for MistralCache since it only contains primitive types
unsafe impl Send for MistralCache {}

impl MistralCache {
    fn new() -> Self {
        Self { seqlen_offset: 0 }
    }

    fn increment_offset(&mut self) {
        self.seqlen_offset += 1;
        tracing::debug!("Cache seqlen_offset incremented to {}", self.seqlen_offset);
    }
}

#[derive(Debug)]
pub struct MistralWithConfig {
    model: RefCell<Mistral>,
}

impl MistralWithConfig {
    fn get_head_dim(hidden_size: usize, num_attention_heads: usize) -> usize {
        let head_dim = hidden_size / num_attention_heads;
        assert!(
            head_dim * num_attention_heads == hidden_size,
            "hidden_size must be divisible by num_attention_heads"
        );
        head_dim
    }
}

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

impl ModelInitializer for MistralWithConfig {
    type Config = ConfigFile;
    type Cache = MistralCache;

    fn initialize_model(
        config: &Self::Config,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<(Self, Self::Cache)> {
        let mistral_config = MistralConfig::from(config.clone());
        let head_dim = Self::get_head_dim(mistral_config.hidden_size, mistral_config.num_attention_heads);
        tracing::debug!(
            "Model config: hidden_size={}, layers={}, heads={}",
            mistral_config.hidden_size, mistral_config.num_hidden_layers, mistral_config.num_attention_heads
        );

        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        tracing::info!("Initializing model");
        let model = Mistral::new(&mistral_config, vb)?;

        Ok((Self { model: RefCell::new(model) }, MistralCache::new()))
    }

    fn initialize_cache(_device: &Device, _dtype: DType) -> Result<Self::Cache> {
        Ok(MistralCache::new())
    }

    fn forward(
        &self,
        input: &Tensor,
        _pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = input.dims2()?;
        tracing::debug!(
            "Forward pass input shape: batch_size={}, seq_len={}, seqlen_offset={}",
            batch_size,
            seq_len,
            cache.seqlen_offset
        );
        // For the first token in a new conversation, reset the cache
        if cache.seqlen_offset == 0 {
            self.model.borrow_mut().clear_kv_cache();
        }
        // Use RefCell to get mutable access and convert candle_core::Error to anyhow::Error
        // Ok(self.model.borrow_mut().forward(input, pos)?)
        let output: Tensor = self.model.borrow_mut().forward(input, cache.seqlen_offset)?;
        cache.increment_offset();
        Ok(output)
    }
}