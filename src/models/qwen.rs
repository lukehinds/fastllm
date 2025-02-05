use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, Activation};
use candle_transformers::models::qwen2::{Config as QwenConfig, ModelForCausalLM as Qwen};
use std::cell::RefCell;

// Wrapper type for cache management
#[derive(Debug)]
pub struct QwenCache {
    seqlen_offset: usize,
}

// Implement Send for QwenCache since it only contains primitive types
unsafe impl Send for QwenCache {}

impl QwenCache {
    fn new() -> Self {
        Self { seqlen_offset: 0 }
    }

    fn increment_offset(&mut self) {
        self.seqlen_offset += 1;
        tracing::debug!("Cache seqlen_offset incremented to {}", self.seqlen_offset);
    }
}

#[derive(Debug)]
pub struct QwenWithConfig {
    model: RefCell<Qwen>,
}

impl QwenWithConfig {
    fn get_head_dim(hidden_size: usize, num_attention_heads: usize) -> usize {
        let head_dim = hidden_size / num_attention_heads;
        assert!(
            head_dim * num_attention_heads == hidden_size,
            "hidden_size must be divisible by num_attention_heads"
        );
        head_dim
    }
}
use serde::Deserialize;
use std::collections::HashMap;
use super::model_initializer::ModelInitializer;
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
    pub max_position_embeddings: usize,  // Required for Qwen
    pub sliding_window: usize,  // Required for Qwen
    pub max_window_layers: Option<usize>,
    pub torch_dtype: Option<String>,
}

impl From<ConfigFile> for QwenConfig {
    fn from(cf: ConfigFile) -> Self {
        // Validate head dimensions
        QwenWithConfig::get_head_dim(cf.hidden_size, cf.num_attention_heads);
        Self {
            hidden_size: cf.hidden_size,
            intermediate_size: cf.intermediate_size,
            vocab_size: cf.vocab_size,
            num_hidden_layers: cf.num_hidden_layers,
            num_attention_heads: cf.num_attention_heads,
            num_key_value_heads: cf.num_key_value_heads.unwrap_or(cf.num_attention_heads),
            rms_norm_eps: cf.rms_norm_eps,
            rope_theta: cf.rope_theta.unwrap_or(10000.0),
            max_position_embeddings: cf.max_position_embeddings,
            sliding_window: cf.sliding_window,
            max_window_layers: cf.max_window_layers.unwrap_or(1),
            tie_word_embeddings: false,
            use_sliding_window: true,  // Always true since sliding_window is required
            hidden_act: Activation::Silu,
        }
    }
}

impl ModelInitializer for QwenWithConfig {
    type Config = ConfigFile;
    type Cache = QwenCache;

    fn initialize_model(
        config: &Self::Config,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<(Self, Self::Cache)> {
        let qwen_config = QwenConfig::from(config.clone());
        let head_dim = Self::get_head_dim(qwen_config.hidden_size, qwen_config.num_attention_heads);
        tracing::debug!(
            "Model config: hidden_size={}, layers={}, heads={}, head_dim={}, max_pos={}",
            qwen_config.hidden_size,
            qwen_config.num_hidden_layers,
            qwen_config.num_attention_heads,
            head_dim,
            qwen_config.max_position_embeddings
        );

        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        tracing::info!("Initializing model");
        let model = Qwen::new(&qwen_config, vb)?;

        Ok((Self { model: RefCell::new(model) }, QwenCache::new()))
    }

    fn initialize_cache(_device: &Device, _dtype: DType) -> Result<Self::Cache> {
        Ok(QwenCache::new())
    }

    fn forward(
        &self,
        input: &Tensor,
        _pos: usize,  // Position is handled internally by Qwen
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        // Ensure input tensor has correct shape before forward pass
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
        let output = self.model.borrow_mut().forward(input, cache.seqlen_offset)?;
        cache.increment_offset();
        Ok(output)
    }
}