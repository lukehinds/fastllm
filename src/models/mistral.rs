use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mistral::{Config as MistralConfig, Model as Mistral};
use candle_nn::Activation;
use serde::Deserialize;
use std::collections::HashMap;
use std::cell::RefCell;

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
    model: RefCell<Mistral>,
    config: MistralConfig,
}

impl ModelInitializer for MistralWithConfig {
    type Config = ConfigFile;
    type Cache = ();

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
        tracing::info!("Initializing model");
        let model = Mistral::new(&mistral_config, vb)?;

        Ok((Self { model: RefCell::new(model), config: mistral_config }, ()))
    }

    fn initialize_cache(_device: &Device, _dtype: DType) -> Result<Self::Cache> {
        Ok(())
    }

    fn forward(
        &self,
        input: &Tensor,
        pos: usize,
        _cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        // Use RefCell to get mutable access and convert candle_core::Error to anyhow::Error
        Ok(self.model.borrow_mut().forward(input, pos)?)
    }
}