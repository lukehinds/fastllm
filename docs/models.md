# Adding New Models

This guide explains how to add new models to the FastLLM inference API using our standardized model configuration system.

## Model Configuration System

The system consists of three main components:

1. **Base Configuration** (`BaseModelConfig`)
2. **Model Metadata** (`ModelMetadata`)
3. **Model Registry** (`ModelRegistry`)

### Base Configuration

The `BaseModelConfig` struct provides common configuration parameters used across different model architectures:

```rust
pub struct BaseModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>,
    pub max_position_embeddings: Option<usize>,
    pub sliding_window: Option<usize>,
    pub torch_dtype: Option<String>,
    pub model_metadata: Option<ModelMetadata>,
    pub quantization_config: Option<QuantizationConfig>,
}
```

### Model Metadata

The `ModelMetadata` struct provides information about model capabilities and requirements:

```rust
pub struct ModelMetadata {
    pub architecture: String,
    pub capabilities: Vec<ModelCapability>,
    pub default_config: HashMap<String, String>,
    pub requirements: ModelRequirements,
}

pub enum ModelCapability {
    Chat,
    Completion,
    Embeddings,
}

pub struct ModelRequirements {
    pub min_gpu_memory: Option<u64>,
    pub min_system_memory: u64,
    pub supports_quantization: bool,
    pub supported_dtypes: Vec<String>,
}
```

## Adding a New Model

To add a new model, follow these steps:

1. Create a new file for your model (e.g., `src/models/your_model.rs`)
2. Implement the required traits:

```rust
use crate::models::{BaseModelConfig, ModelConfig, TextGeneration};

pub struct YourModelWithConfig {
    config: BaseModelConfig,
    tokenizer: Tokenizer,
}

impl ModelConfig for YourModelWithConfig {
    fn base_config(&self) -> &BaseModelConfig {
        &self.config
    }
}

impl TextGeneration for YourModelWithConfig {
    fn forward(&self, input_tokens: &Tensor, cache: Option<&mut KVCache>) -> Result<Tensor> {
        // Implement your model's forward pass
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}
```

3. Register your model in `ModelRegistry`:

```rust
self.register_with_metadata(
    "YourModel",
    Box::new(|model_id, revision, dtype, device| {
        Box::pin(async move {
            let model = crate::models::load_model::<YourModelWithConfig>(
                &model_id, &revision, dtype, &device,
            )
            .await?;
            Ok(ModelWrapper::YourModel(model, model_id))
        })
    }),
    ModelMetadata::new(
        "YourModel",
        vec![ModelCapability::Chat, ModelCapability::Completion],
        ModelRequirements {
            min_gpu_memory: Some(8 * 1024 * 1024 * 1024), // 8GB
            min_system_memory: 16 * 1024 * 1024 * 1024,   // 16GB
            supports_quantization: true,
            supported_dtypes: vec!["float16".into(), "float32".into()],
        },
    )
    .with_default_config("max_sequence_length", "4096")
    .with_default_config("temperature", "0.7"),
);
```

## Model Validation

The `ModelConfig` trait provides default implementations for common validation tasks:

- `validate_head_dimensions`: Ensures proper attention head configuration
- `validate_gqa_config`: Validates grouped-query attention settings
- `validate`: Runs all validation checks

You can override these methods if your model requires custom validation.

## Testing

Always include tests for your model implementation:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_validation() {
        let config = BaseModelConfig {
            // Set up test configuration
        };
        let model = YourModelWithConfig::new(config);
        assert!(model.validate().is_ok());
    }
}
```

## Best Practices

1. **Configuration**: Use the `BaseModelConfig` as your model's foundation
2. **Metadata**: Provide accurate metadata about your model's capabilities and requirements
3. **Validation**: Implement proper validation to catch configuration errors early
4. **Testing**: Write comprehensive tests for your model's configuration and behavior
5. **Documentation**: Document any model-specific features or requirements

## Example Models

See existing model implementations for reference:
- `src/models/llama.rs`
- `src/models/mistral.rs`
- `src/models/qwen.rs`