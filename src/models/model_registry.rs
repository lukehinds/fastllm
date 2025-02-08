use std::collections::HashMap;
use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use crate::models::{
    ModelWrapper,
    qwen::QwenWithConfig,
    mistral::MistralWithConfig,
    llama::LlamaWithConfig,
    embeddings::MiniLMModel,
};

type ModelInitFn = Box<dyn Fn(String, String, DType, Device) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ModelWrapper>> + Send>> + Send + Sync>;

pub struct ModelRegistry {
    models: HashMap<String, ModelInitFn>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
        };
        registry.register_defaults();
        registry
    }

    fn register_defaults(&mut self) {
        // Register Qwen models
        self.register("Qwen", Box::new(|model_id: String, revision: String, dtype: DType, device: Device| {
            Box::pin(async move {
                let model = crate::models::load_model::<QwenWithConfig>(&model_id, &revision, dtype, &device).await?;
                Ok(ModelWrapper::Qwen(model, model_id))
            })
        }));

        // Register Mistral models
        self.register("Mistral", Box::new(|model_id: String, revision: String, dtype: DType, device: Device| {
            Box::pin(async move {
                let model = crate::models::load_model::<MistralWithConfig>(&model_id, &revision, dtype, &device).await?;
                Ok(ModelWrapper::Mistral(model, model_id))
            })
        }));

        // Register Llama models
        self.register("Llama", Box::new(|model_id: String, revision: String, dtype: DType, device: Device| {
            Box::pin(async move {
                let model = crate::models::load_model::<LlamaWithConfig>(&model_id, &revision, dtype, &device).await?;
                Ok(ModelWrapper::Llama(model, model_id))
            })
        }));

        // Register embedding models
        self.register("all-MiniLM-L6-v2", Box::new(|model_id: String, _revision: String, _dtype: DType, _device: Device| {
            Box::pin(async move {
                let model = MiniLMModel::new(&model_id)?;
                Ok(ModelWrapper::Embedding(Box::new(model)))
            })
        }));
    }

    pub fn register<S: Into<String>>(&mut self, model_type: S, init_fn: ModelInitFn) {
        self.models.insert(model_type.into(), init_fn);
    }

    pub async fn create_model(&self, model_id: &str, revision: &str, dtype: DType, device: &Device) -> Result<ModelWrapper> {
        for (model_type, init_fn) in &self.models {
            if model_id.contains(model_type) {
                return init_fn(model_id.to_string(), revision.to_string(), dtype, device.clone()).await;
            }
        }
        Err(anyhow!("Unsupported model: {}", model_id))
    }
} 