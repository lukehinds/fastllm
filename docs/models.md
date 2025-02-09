# Adding New Models

This guide explains how to add new models to the FastLLM inference API using our standardized model configuration system.

## Model Architecture Detection

FastLLM uses a robust architecture detection system that relies on the model's `config.json` file. Each model must implement the `ModelArchitecture` trait:

```rust
pub trait ModelArchitecture {
    fn get_family() -> &'static str;
    fn supports_architecture(architecture: &str) -> bool;
}
```

This trait enables:
1. Automatic detection of model architectures from HuggingFace config files
2. Mapping of architectures to model families
3. Validation that a model implementation supports specific architectures

### Supported Families and Architectures

Currently supported model families and their architectures:

- **Llama Family**
  - LlamaForCausalLM
  
- **Mistral Family**
  - MistralForCausalLM
  
- **Qwen Family**
  - Qwen2ForCausalLM
  - Qwen2_5_VLForConditionalGeneration
  
- **BERT Family**
  - BertModel
  - RobertaModel
  - DebertaModel

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
use crate::models::{BaseModelConfig, ModelConfig, TextGeneration, ModelArchitecture};

pub struct YourModelWithConfig {
    config: BaseModelConfig,
    tokenizer: Tokenizer,
}

impl ModelArchitecture for YourModelWithConfig {
    fn get_family() -> &'static str {
        "YourFamily"
    }

    fn supports_architecture(architecture: &str) -> bool {
        matches!(architecture, "YourModelForCausalLM" | "YourModelForSequenceClassification")
    }
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
self.register(
    YourModelWithConfig::get_family(),
    Box::new(|model_id, revision, dtype, device| {
        Box::pin(async move {
            let model = crate::models::load_model::<YourModelWithConfig>(
                &model_id, &revision, dtype, &device,
            )
            .await?;
            Ok(ModelWrapper::YourModel(model, model_id))
        })
    }),
);
```

## Testing

Include tests for both model implementation and architecture detection:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_support() {
        assert!(YourModelWithConfig::supports_architecture("YourModelForCausalLM"));
        assert!(!YourModelWithConfig::supports_architecture("UnsupportedArchitecture"));
    }

    #[tokio::test]
    async fn test_model_loading() {
        // Test that your model loads correctly with its config.json
    }
}
```

## Best Practices

1. **Architecture Detection**: 
   - Implement `ModelArchitecture` trait for reliable model type detection
   - Use specific architecture names from HuggingFace's model hub
   - Test architecture detection with real model configs

2. **Family Mapping**:
   - Group related architectures under the same family
   - Use clear, consistent family names
   - Document supported architectures for each family

3. **Configuration**: Use the `BaseModelConfig` as your model's foundation
4. **Metadata**: Provide accurate metadata about your model's capabilities and requirements
5. **Validation**: Implement proper validation to catch configuration errors early
6. **Testing**: Write comprehensive tests for your model's configuration and behavior
7. **Documentation**: Document any model-specific features or requirements

## Example Models

See existing model implementations for reference:
- `src/models/llama.rs` - LLaMA family models
- `src/models/mistral.rs` - Mistral family models
- `src/models/qwen.rs` - Qwen family models
- `src/models/embeddings.rs` - BERT family models