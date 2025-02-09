use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{
    Cache, Config as LlamaConfig, Llama as CandelLlama, LlamaEosToks,
};
use std::any::Any;

pub type Llama = CandelLlama;
use serde::Deserialize;
use std::collections::HashMap;

use super::cache::ModelCache;
use super::model_initializer::{ModelArchitecture, ModelInitializer};

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
            eos_token_id: Some(LlamaEosToks::Single(2)), // Common EOS token ID for LLaMA models
            bos_token_id: Some(1),
            rope_scaling: None,
            tie_word_embeddings: false,
            max_position_embeddings: cf.max_position_embeddings.unwrap_or(4096),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlamaWithConfig {
    model: Llama,
}

// Implement Send and Sync since Llama is thread-safe
unsafe impl Send for LlamaWithConfig {}
unsafe impl Sync for LlamaWithConfig {}

#[derive(Debug)]
pub struct LlamaCache {
    inner: Cache,
    seqlen_offset: usize,
}

impl LlamaCache {
    pub fn new(inner: Cache) -> Self {
        Self {
            inner,
            seqlen_offset: 0,
        }
    }
}

impl ModelCache for LlamaCache {
    fn increment_offset(&mut self) {
        self.seqlen_offset += 1;
    }

    fn reset(&mut self) {
        self.seqlen_offset = 0;
    }

    fn get_offset(&self) -> usize {
        self.seqlen_offset
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ModelInitializer for LlamaWithConfig {
    type Config = ConfigFile;
    type Cache = LlamaCache;

    fn initialize_model(
        config: &Self::Config,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<(Self, Self::Cache)> {
        let llama_config = LlamaConfig::from(config.clone());
        tracing::debug!(
            "Model config: hidden_size={}, layers={}, heads={}",
            llama_config.hidden_size,
            llama_config.num_hidden_layers,
            llama_config.num_attention_heads
        );

        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        tracing::info!("Initializing model cache");
        let inner_cache = Cache::new(true, dtype, &llama_config, device)
            .context("Failed to initialize model cache")?;
        let cache = LlamaCache::new(inner_cache);

        tracing::info!("Initializing model");
        let model = Llama::load(vb, &llama_config).context("Failed to initialize model")?;

        Ok((Self { model }, cache))
    }

    fn initialize_cache(device: &Device, dtype: DType) -> Result<Self::Cache> {
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
        let inner_cache = Cache::new(true, dtype, &default_config, device)
            .context("Failed to initialize model cache")?;
        Ok(LlamaCache::new(inner_cache))
    }

    fn forward(&self, input: &Tensor, pos: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        Ok(self.model.forward(input, pos, &mut cache.inner)?)
    }
}

impl ModelArchitecture for LlamaWithConfig {
    fn get_family() -> &'static str {
        "Llama"
    }

    fn supports_architecture(architecture: &str) -> bool {
        architecture == "LlamaForCausalLM"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_llama_cache_operations() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let config = LlamaConfig {
            hidden_size: 512,
            intermediate_size: 1024,
            vocab_size: 1000,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
            use_flash_attn: false,
            eos_token_id: Some(LlamaEosToks::Single(2)),
            bos_token_id: Some(1),
            rope_scaling: None,
            tie_word_embeddings: false,
        };

        let inner_cache = Cache::new(true, dtype, &config, &device).unwrap();
        let mut cache = LlamaCache::new(inner_cache);

        assert_eq!(cache.get_offset(), 0, "Initial offset should be 0");

        cache.increment_offset();
        assert_eq!(cache.get_offset(), 1, "Offset should be 1 after increment");

        cache.increment_offset();
        assert_eq!(
            cache.get_offset(),
            2,
            "Offset should be 2 after second increment"
        );

        cache.reset();
        assert_eq!(cache.get_offset(), 0, "Offset should be 0 after reset");
    }

    #[test]
    fn test_llama_config_conversion() {
        let config_file = ConfigFile {
            hidden_size: 512,
            intermediate_size: 1024,
            vocab_size: 1000,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: Some(8),
            rms_norm_eps: 1e-5,
            rope_theta: Some(10000.0),
            max_position_embeddings: Some(2048),
            torch_dtype: None,
        };

        let llama_config = LlamaConfig::from(config_file);

        assert_eq!(llama_config.hidden_size, 512);
        assert_eq!(llama_config.intermediate_size, 1024);
        assert_eq!(llama_config.vocab_size, 1000);
        assert_eq!(llama_config.num_hidden_layers, 2);
        assert_eq!(llama_config.num_attention_heads, 8);
        assert_eq!(llama_config.num_key_value_heads, 8);
        assert_eq!(llama_config.max_position_embeddings, 2048);
        assert_eq!(llama_config.rope_theta, 10000.0);
    }

    #[test]
    fn test_llama_cache_as_any() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let config = LlamaConfig {
            hidden_size: 512,
            intermediate_size: 1024,
            vocab_size: 1000,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
            use_flash_attn: false,
            eos_token_id: Some(LlamaEosToks::Single(2)),
            bos_token_id: Some(1),
            rope_scaling: None,
            tie_word_embeddings: false,
        };

        let inner_cache = Cache::new(true, dtype, &config, &device).unwrap();
        let mut cache = LlamaCache::new(inner_cache);

        let any_cache = cache.as_any_mut();
        assert!(
            any_cache.downcast_mut::<LlamaCache>().is_some(),
            "Should be able to downcast to LlamaCache"
        );
        assert!(
            any_cache.downcast_mut::<String>().is_none(),
            "Should not be able to downcast to wrong type"
        );
    }
}
