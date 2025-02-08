use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mistral::{Config as MistralConfig, Model as Mistral};
use candle_nn::Activation;
use serde::Deserialize;
use std::collections::HashMap;
use std::cell::RefCell;
use std::any::Any;

use super::model_initializer::ModelInitializer;
use super::cache::ModelCache;


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
}

impl ModelCache for MistralCache {
    fn increment_offset(&mut self) {
        self.seqlen_offset += 1;
        tracing::debug!("Cache seqlen_offset incremented to {}", self.seqlen_offset);
    }
    
    fn reset(&mut self) {
        self.seqlen_offset = 0;
        tracing::debug!("Cache reset");
    }
    
    fn get_offset(&self) -> usize {
        self.seqlen_offset
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Debug)]
pub struct MistralWithConfig {
    model: RefCell<Mistral>,
}

// Implement Clone manually since RefCell doesn't implement Clone
impl Clone for MistralWithConfig {
    fn clone(&self) -> Self {
        Self {
            model: RefCell::new(self.model.borrow().clone()),
        }
    }
}

// Implement Send and Sync since Mistral is thread-safe
unsafe impl Send for MistralWithConfig {}
unsafe impl Sync for MistralWithConfig {}

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
        // Calculate head dimensions
        let head_dim = MistralWithConfig::get_head_dim(cf.hidden_size, cf.num_attention_heads);
        let num_key_value_heads = cf.num_key_value_heads.unwrap_or(cf.num_attention_heads);

        tracing::debug!(
            "Initializing Mistral config with hidden_size={}, head_dim={}, num_attention_heads={}, num_kv_heads={}, rope_theta={}",
            cf.hidden_size,
            head_dim,
            cf.num_attention_heads,
            num_key_value_heads,
            cf.rope_theta.unwrap_or(10000.0)
        );

        // Validate GQA configuration
        assert!(
            cf.num_attention_heads % num_key_value_heads == 0,
            "num_attention_heads must be divisible by num_key_value_heads"
        );

        // Validate head dimensions
        assert!(
            head_dim * cf.num_attention_heads == cf.hidden_size,
            "head_dim ({}) * num_attention_heads ({}) must equal hidden_size ({})",
            head_dim,
            cf.num_attention_heads,
            cf.hidden_size
        );

        // Validate RoPE dimensions
        assert!(
            head_dim % 2 == 0,
            "head_dim must be even for RoPE embeddings"
        );

        let config = Self {
            hidden_size: cf.hidden_size,
            intermediate_size: cf.intermediate_size,
            vocab_size: cf.vocab_size,
            num_hidden_layers: cf.num_hidden_layers,
            num_attention_heads: cf.num_attention_heads,
            num_key_value_heads: num_key_value_heads,
            rms_norm_eps: cf.rms_norm_eps,
            rope_theta: cf.rope_theta.unwrap_or(10000.0),
            max_position_embeddings: cf.max_position_embeddings.unwrap_or(32768),
            sliding_window: Some(cf.sliding_window.unwrap_or(4096)),
            use_flash_attn: false,
            head_dim: Some(head_dim),
            hidden_act: Activation::Silu,
        };

        tracing::debug!(
            "RoPE dimensions: head_dim={}, rope_dim={}, max_position_embeddings={}",
            head_dim,
            head_dim / 2,
            config.max_position_embeddings
        );

        config
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
            "Model dimensions: hidden_size={}, head_dim={}, num_heads={}, num_kv_heads={}, max_pos={}",
            mistral_config.hidden_size,
            head_dim,
            mistral_config.num_attention_heads,
            mistral_config.num_key_value_heads,
            mistral_config.max_position_embeddings,
        );

        // Validate RoPE dimensions
        let rope_dim = head_dim / 2;
        tracing::debug!(
            "RoPE dimensions for attention: head_dim={}, rope_dim={}, theta={}",
            head_dim,
            rope_dim,
            mistral_config.rope_theta
        );

        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        tracing::info!("Initializing model with dtype={:?}", dtype);
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
            "Forward pass: batch_size={}, seq_len={}, seqlen_offset={}, input_shape={:?}, input_dtype={:?}",
            batch_size,
            seq_len,
            cache.seqlen_offset,
            input.shape(),
            input.dtype()
        );

        // For the first token in a new conversation, reset the cache
        if cache.seqlen_offset == 0 {
            tracing::debug!("Resetting KV cache at start of conversation");
            self.model.borrow_mut().clear_kv_cache();
        }

        let output = self.model.borrow_mut().forward(input, cache.seqlen_offset)?;
        tracing::debug!(
            "Forward pass complete: output_shape={:?}, output_dtype={:?}, seqlen_offset={}",
            output.shape(),
            output.dtype(),
            cache.seqlen_offset
        );

        cache.increment_offset();
        Ok(output)
    }
}