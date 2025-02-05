use anyhow::{Result, Context};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as QwenConfig, ModelForCausalLM as Qwen};
use candle_transformers::generation::LogitsProcessor;
use std::cell::RefCell;
use std::collections::HashMap;

use super::model_initializer::ModelInitializer;
use super::{ModelCache, CommonCache, BaseModelConfig, ModelConfigValidation, ModelForward, ModelGeneration};

#[derive(Debug)]
pub struct QwenWithConfig {
    model: RefCell<Qwen>,
}

impl From<BaseModelConfig> for QwenConfig {
    fn from(base: BaseModelConfig) -> Self {
        let _ = base.validate_head_dimensions()
            .expect("Invalid head dimensions");
        
        base.validate_gqa_config()
            .expect("Invalid GQA configuration");

        Self {
            hidden_size: base.hidden_size,
            intermediate_size: base.intermediate_size,
            vocab_size: base.vocab_size,
            num_hidden_layers: base.num_hidden_layers,
            num_attention_heads: base.num_attention_heads,
            num_key_value_heads: base.num_key_value_heads
                .unwrap_or(base.num_attention_heads),
            rms_norm_eps: base.rms_norm_eps,
            rope_theta: base.rope_theta.unwrap_or(10000.0),
            max_position_embeddings: base.max_position_embeddings
                .unwrap_or(32768),
            sliding_window: base.sliding_window.unwrap_or(4096),
            max_window_layers: 1,
            tie_word_embeddings: false,
            use_sliding_window: true,
            hidden_act: BaseModelConfig::get_activation(),
        }
    }
}

impl ModelForward for QwenWithConfig {
    fn forward_pass(&self, input: &Tensor, cache: &mut dyn ModelCache) -> Result<Tensor> {
        let (batch_size, seq_len) = input.dims2()?;
        tracing::debug!(
            "Forward pass input shape: batch_size={}, seq_len={}, seqlen_offset={}",
            batch_size,
            seq_len,
            cache.get_offset()
        );

        if cache.get_offset() == 0 {
            self.clear_cache();
        }

        let output = self.model.borrow_mut().forward(input, cache.get_offset())?;
        cache.increment_offset();
        Ok(output)
    }

    fn clear_cache(&self) {
        self.model.borrow_mut().clear_kv_cache();
        tracing::debug!("KV cache cleared");
    }
}

impl ModelInitializer for QwenWithConfig {
    type Config = BaseModelConfig;
    type Cache = CommonCache;

    fn initialize_model(
        config: &Self::Config,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<(Self, Self::Cache)> {
        let qwen_config = QwenConfig::from(config.clone());
        
        tracing::debug!(
            "Model config: hidden_size={}, layers={}, heads={}", 
            qwen_config.hidden_size,
            qwen_config.num_hidden_layers,
            qwen_config.num_attention_heads,
        );

        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let model = Qwen::new(&qwen_config, vb)?;

        Ok((Self { 
            model: RefCell::new(model),
        }, CommonCache::new()))
    }

    fn initialize_cache(_device: &Device, _dtype: DType) -> Result<Self::Cache> {
        Ok(CommonCache::new())
    }

    fn forward(
        &self,
        input: &Tensor,
        _pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        self.forward_pass(input, cache)
    }
}

#[allow(unused)]
impl ModelGeneration for QwenWithConfig {
    fn sample_next_token(&self, logits: &Tensor, temperature: f32) -> Result<usize> {
        let mut logits_processor = LogitsProcessor::new(Default::default(), Some(temperature as f64), None);
        logits_processor.sample(logits)
            .map(|x| x as usize)
            .context("Failed to sample next token")
    }

    fn is_eos_token(&self, token_id: usize) -> bool {
        token_id == self.get_eos_token_id().unwrap_or(2)
    }

    fn get_eos_token_id(&self) -> Option<usize> {
        Some(2)
    }
}