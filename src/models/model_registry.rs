use crate::models::{
    embeddings::MiniLMModel, llama::LlamaWithConfig, mistral::MistralWithConfig,
    qwen::QwenWithConfig, ModelWrapper,
};
use anyhow::{anyhow, Result};
use candle_core::{DType, Device};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct ModelConfig {
    architectures: Vec<String>,
}

type ModelInitFn = Box<
    dyn Fn(
            String,
            String,
            DType,
            Device,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ModelWrapper>> + Send>>
        + Send
        + Sync,
>;

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
        self.register(
            "Qwen",
            Box::new(
                |model_id: String, revision: String, dtype: DType, device: Device| {
                    Box::pin(async move {
                        let model = crate::models::load_model::<QwenWithConfig>(
                            &model_id, &revision, dtype, &device,
                        )
                        .await?;
                        Ok(ModelWrapper::Qwen(model, model_id))
                    })
                },
            ),
        );

        // Register Mistral models
        self.register(
            "Mistral",
            Box::new(
                |model_id: String, revision: String, dtype: DType, device: Device| {
                    Box::pin(async move {
                        let model = crate::models::load_model::<MistralWithConfig>(
                            &model_id, &revision, dtype, &device,
                        )
                        .await?;
                        Ok(ModelWrapper::Mistral(model, model_id))
                    })
                },
            ),
        );

        // Register Llama models
        self.register(
            "Llama",
            Box::new(
                |model_id: String, revision: String, dtype: DType, device: Device| {
                    Box::pin(async move {
                        let model = crate::models::load_model::<LlamaWithConfig>(
                            &model_id, &revision, dtype, &device,
                        )
                        .await?;
                        Ok(ModelWrapper::Llama(model, model_id))
                    })
                },
            ),
        );

        // Register BERT family models
        self.register(
            "bert",
            Box::new(
                |model_id: String, _revision: String, _dtype: DType, _device: Device| {
                    Box::pin(async move {
                        let model = MiniLMModel::new(&model_id)?;
                        Ok(ModelWrapper::Embedding(Box::new(model)))
                    })
                },
            ),
        );
    }

    pub fn register<S: Into<String>>(&mut self, model_type: S, init_fn: ModelInitFn) {
        self.models.insert(model_type.into(), init_fn);
    }

    async fn get_model_architecture(&self, model_id: &str, revision: &str) -> Result<String> {
        use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

        let api = ApiBuilder::new()
            .with_token(std::env::var("HF_TOKEN").ok())
            .build()?;

        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        let config_path = repo.get("config.json")?;
        let config_content = std::fs::read_to_string(config_path)?;
        let config: ModelConfig = serde_json::from_str(&config_content)?;

        config
            .architectures
            .first()
            .ok_or_else(|| anyhow!("No architecture found in config.json"))
            .map(|s| s.to_string())
    }

    fn get_family_from_architecture(&self, architecture: &str) -> Option<&str> {
        match architecture {
            arch if arch.contains("Llama") => Some("Llama"),
            arch if arch.contains("Mistral") => Some("Mistral"),
            arch if arch.contains("Qwen") => Some("Qwen"),
            arch if arch.contains("Bert")
                || arch.contains("Roberta")
                || arch.contains("Deberta") =>
            {
                Some("bert")
            }
            _ => None,
        }
    }

    pub async fn create_model(
        &self,
        model_id: &str,
        revision: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<ModelWrapper> {
        // First get the model's architecture from config.json
        let architecture = self.get_model_architecture(model_id, revision).await?;
        tracing::info!("Detected model architecture: {}", architecture);

        // Map the architecture to a family
        let family = self
            .get_family_from_architecture(&architecture)
            .ok_or_else(|| anyhow!("Unsupported model architecture: {}", architecture))?;
        tracing::info!("Mapped to model family: {}", family);

        // Get the initialization function for this family
        if let Some(init_fn) = self.models.get(family) {
            return init_fn(
                model_id.to_string(),
                revision.to_string(),
                dtype,
                device.clone(),
            )
            .await;
        }

        Err(anyhow!(
            "No implementation found for architecture {} (family {})",
            architecture,
            family
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_family_mapping() {
        let registry = ModelRegistry::new();

        // Test Llama family
        assert_eq!(
            registry.get_family_from_architecture("LlamaForCausalLM"),
            Some("Llama"),
            "Should map LlamaForCausalLM to Llama family"
        );

        // Test Mistral family
        assert_eq!(
            registry.get_family_from_architecture("MistralForCausalLM"),
            Some("Mistral"),
            "Should map MistralForCausalLM to Mistral family"
        );

        // Test Qwen family
        assert_eq!(
            registry.get_family_from_architecture("Qwen2ForCausalLM"),
            Some("Qwen"),
            "Should map Qwen2ForCausalLM to Qwen family"
        );
        assert_eq!(
            registry.get_family_from_architecture("Qwen2_5_VLForConditionalGeneration"),
            Some("Qwen"),
            "Should map Qwen2_5_VLForConditionalGeneration to Qwen family"
        );

        // Test BERT family
        assert_eq!(
            registry.get_family_from_architecture("BertModel"),
            Some("bert"),
            "Should map BertModel to bert family"
        );
        assert_eq!(
            registry.get_family_from_architecture("RobertaModel"),
            Some("bert"),
            "Should map RobertaModel to bert family"
        );
        assert_eq!(
            registry.get_family_from_architecture("DebertaModel"),
            Some("bert"),
            "Should map DebertaModel to bert family"
        );

        // Test unsupported architecture
        assert_eq!(
            registry.get_family_from_architecture("UnsupportedArchitecture"),
            None,
            "Should return None for unsupported architectures"
        );
    }

    #[tokio::test]
    async fn test_model_architecture_detection() -> Result<()> {
        let registry = ModelRegistry::new();

        // Test TinyLlama
        let architecture = registry
            .get_model_architecture("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "main")
            .await?;
        assert_eq!(
            architecture, "LlamaForCausalLM",
            "TinyLlama should report LlamaForCausalLM architecture"
        );

        // Test Mistral
        let architecture = registry
            .get_model_architecture("mistralai/Mistral-7B-v0.1", "main")
            .await?;
        assert_eq!(
            architecture, "MistralForCausalLM",
            "Mistral should report MistralForCausalLM architecture"
        );

        // Test BERT
        let architecture = registry
            .get_model_architecture("sentence-transformers/all-MiniLM-L6-v2", "main")
            .await?;
        assert_eq!(
            architecture, "BertModel",
            "MiniLM should report BertModel architecture"
        );

        Ok(())
    }

    #[test]
    fn test_registry_initialization() {
        let registry = ModelRegistry::new();

        // Test that all expected families are registered
        assert!(
            registry.models.contains_key("Llama"),
            "Llama family should be registered"
        );
        assert!(
            registry.models.contains_key("Mistral"),
            "Mistral family should be registered"
        );
        assert!(
            registry.models.contains_key("Qwen"),
            "Qwen family should be registered"
        );
        assert!(
            registry.models.contains_key("bert"),
            "BERT family should be registered"
        );
    }
}
