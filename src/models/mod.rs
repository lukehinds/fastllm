use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
#[allow(unused_imports)]
use std::collections::HashMap;

pub mod llama;
pub mod mistral;
pub mod qwen;
pub mod model_initializer;
pub mod embeddings;

use crate::providers::huggingface;
pub use huggingface::load_model;
pub use model_initializer::ModelInitializer;

use llama::LlamaWithConfig;
use qwen::QwenWithConfig;
use mistral::MistralWithConfig;

mod cache;
mod config;
mod traits;

pub use cache::{ModelCache, CommonCache};
pub use config::{BaseModelConfig, ModelConfigValidation};
pub use traits::{ModelForward, ModelGeneration};
use embeddings::{EmbeddingModel, EmbeddingOutput};

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct MockLlamaWithConfig {}

#[cfg(test)]
impl ModelInitializer for MockLlamaWithConfig {
    type Config = BaseModelConfig;
    type Cache = CommonCache;

    fn initialize_model(
        _config: &Self::Config,
        _tensors: HashMap<String, Tensor>,
        _dtype: DType,
        _device: &Device,
    ) -> Result<(Self, Self::Cache)> {
        Ok((Self {}, CommonCache::new()))
    }

    fn initialize_cache(_device: &Device, _dtype: DType) -> Result<Self::Cache> {
        Ok(CommonCache::new())
    }

    fn forward(
        &self,
        _input: &Tensor,
        _pos: usize,
        _cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        unimplemented!()
    }
}

pub enum ModelWrapper {
    Llama(Model<LlamaWithConfig>, String),
    Qwen(Model<QwenWithConfig>, String),
    Mistral(Model<MistralWithConfig>, String),
    Embedding(Box<dyn EmbeddingModel + Send>),
    #[cfg(test)]
    Test(Model<MockLlamaWithConfig>, String),  // Now MockLlamaWithConfig is in scope
}

impl std::fmt::Debug for ModelWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama(_, id) => write!(f, "ModelWrapper::Llama({})", id),
            Self::Qwen(_, id) => write!(f, "ModelWrapper::Qwen({})", id),
            Self::Mistral(_, id) => write!(f, "ModelWrapper::Mistral({})", id),
            Self::Embedding(model) => write!(f, "ModelWrapper::Embedding({})", model.model_id()),
            #[cfg(test)]
            Self::Test(_, id) => write!(f, "ModelWrapper::Test({})", id),
        }
    }
}

impl ModelWrapper {
    pub fn model_id(&self) -> String {
        match self {
            #[cfg(test)]
            Self::Test(_, id) => id.clone(),
            Self::Llama(_, id) => id.clone(),
            Self::Qwen(_, id) => id.clone(),
            Self::Mistral(_, id) => id.clone(),
            Self::Embedding(model) => model.model_id(),
        }
    }

    pub fn embedding_size(&self) -> usize {
        match self {
            #[cfg(test)]
            Self::Test(_, _) => panic!("Chat models do not support embeddings"),
            Self::Llama(_, _) | Self::Qwen(_, _) | Self::Mistral(_, _) => 
                panic!("Chat models do not support embeddings"),
            Self::Embedding(model) => model.embedding_size(),
        }
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        match self {
            #[cfg(test)]
            Self::Test(model, _) => model.generate(prompt, max_tokens, temperature),
            Self::Llama(model, _) => model.generate(prompt, max_tokens, temperature),
            Self::Qwen(model, _) => model.generate(prompt, max_tokens, temperature),
            Self::Mistral(model, _) => model.generate(prompt, max_tokens, temperature),
            Self::Embedding(_) => Err(anyhow::anyhow!("Embedding models do not support text generation")),
        }
    }

    pub fn embed(&self, text: &str) -> Result<EmbeddingOutput> {
        match self {
            #[cfg(test)]
            Self::Test(_, _) => Err(anyhow::anyhow!("Chat models do not support embeddings")),
            Self::Llama(_, _) | Self::Qwen(_, _) | Self::Mistral(_, _) => 
                Err(anyhow::anyhow!("Chat models do not support embeddings")),
            Self::Embedding(model) => model.embed(text),
        }
    }
}

pub struct Model<M: ModelInitializer> {
    tokenizer: Tokenizer,
    model: M,
    device: Device,
    logits_processor: LogitsProcessor,
    cache: M::Cache,
    dtype: DType,
}

impl<M: ModelInitializer> Model<M> {
    pub fn new(
        tokenizer: Tokenizer,
        model: M,
        device: Device,
        cache: M::Cache,
    ) -> Self {
        Self {
            tokenizer,
            model,
            device,
            logits_processor: LogitsProcessor::new(Default::default(), None, None),
            cache,
            dtype: DType::BF16,
        }
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        // Reset the cache before each generation
        self.cache = M::initialize_cache(&self.device, self.dtype)?;

        // Update LogitsProcessor with temperature, converting f32 to f64
        self.logits_processor = LogitsProcessor::new(Default::default(), Some(temperature as f64), None);

        tracing::debug!("Generating response for prompt: {}", prompt);

        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = encoding.get_ids();
        tracing::debug!("Input tokens: {}", input_ids.len());

        // Create input tensor with shape [batch_size=1, seq_len]
        let input_tensor = Tensor::new(input_ids, &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?;
        let input_dims = input_tensor.dims();
        tracing::debug!("Input tensor dims: {:?}", input_dims);

        // Reshape input tensor to [batch_size=1, seq_len]
        let input_tensor = input_tensor.reshape((1, input_dims[0]))
            .map_err(|e| anyhow::anyhow!("Failed to reshape input tensor: {}", e))?;
        tracing::debug!("Reshaped input tensor dims: {:?}", input_tensor.dims());

        let mut output_ids = Vec::new();
        let mut pos = 0;

        // Initial forward pass
        tracing::debug!("Performing initial forward pass");
        let mut logits = self.model.forward(&input_tensor, pos, &mut self.cache)
            .map_err(|e| anyhow::anyhow!("Model forward pass failed: {}", e))?;

        tracing::debug!("Initial logits dims: {:?}", logits.dims());
        pos += input_dims[0];

        // Generate new tokens
        for i in 0..max_tokens {
            tracing::trace!("Generation step {}", i);

            // Get the logits for the last token
            let logits_dims = logits.dims();
            tracing::debug!("Current logits dims: {:?}", logits_dims);

            // Get the logits for the last position
            // The logits tensor has shape [batch_size=1, vocab_size]
            // Reshape logits to [batch_size=1, vocab_size] before sampling
            let last_logits = logits.get(0)?.flatten_all()?;
            tracing::debug!("Last logits dims: {:?}", last_logits.dims());

            // Sample the next token
            let next_token_id = self.logits_processor.sample(&last_logits)
                .map_err(|e| anyhow::anyhow!("Failed to sample next token: {}", e))?;
            tracing::trace!("Generated token ID: {}", next_token_id);

            if let Some(eos_token_id) = self.tokenizer.token_to_id("</s>") {
                if next_token_id == eos_token_id {
                    tracing::debug!("End of sequence token generated");
                    break;
                }
            }

            output_ids.push(next_token_id);

            // Create tensor for the next token with shape [batch_size=1, seq_len=1]
            let next_input = Tensor::new(&[next_token_id], &self.device)
                .map_err(|e| anyhow::anyhow!("Failed to create next token tensor: {}", e))?
                .reshape((1, 1))
                .map_err(|e| anyhow::anyhow!("Failed to reshape next token tensor: {}", e))?;

            logits = self.model.forward(&next_input, pos, &mut self.cache)
                .map_err(|e| anyhow::anyhow!("Model forward pass failed at position {}: {}", pos, e))?;
            pos += 1;
        }

        tracing::debug!("Generated {} tokens", output_ids.len());
        let output = self.tokenizer.decode(&output_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;
        tracing::debug!("Decoded output: {}", output);

        Ok(output)
    }
}

pub fn default_dtype() -> DType {
    DType::BF16  // Default to BF16 since that's what the model expects
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{Tokenizer, models::bpe::BPE};

    // Create a simpler mock tokenizer
    fn create_test_tokenizer() -> Tokenizer {
        // Create a minimal BPE model
        let vocab = std::collections::HashMap::from([
            ("test".into(), 0),
            ("token".into(), 1),
            ("<s>".into(), 2),
            ("</s>".into(), 3),
        ]);
        let merges = vec![];
        let bpe = BPE::new(vocab, merges);
        
        // Create tokenizer with the BPE model
        Tokenizer::new(bpe)
    }

    // Instead of using catch_unwind, let's test the panic directly
    #[test]
    #[should_panic(expected = "Chat models do not support embeddings")]
    fn test_embedding_size_error() {
        let model_id = "test-model";
        let wrapper = ModelWrapper::Test(
            Model::new(
                create_test_tokenizer(),
                MockLlamaWithConfig {},
                Device::Cpu,
                CommonCache::new(),
            ),
            model_id.to_string()
        );

        wrapper.embedding_size();
    }

    #[test]
    fn test_model_wrapper_model_id() {
        let model_id = "test-model";
        let wrapper = ModelWrapper::Test(
            Model::new(
                create_test_tokenizer(),
                MockLlamaWithConfig {},
                Device::Cpu,
                CommonCache::new(),
            ),
            model_id.to_string()
        );

        assert_eq!(wrapper.model_id(), model_id);
    }

    #[test]
    fn test_default_dtype() {
        assert_eq!(default_dtype(), DType::BF16);
    }

    #[test]
    fn test_generate_error_on_embedding_model() {
        let mut wrapper = ModelWrapper::Embedding(Box::new(MockEmbeddingModel {}));
        let result = wrapper.generate("test prompt", 10, 0.7);
        assert!(result.is_err());
    }

    // Mock implementation for testing
    struct MockEmbeddingModel {}

    impl EmbeddingModel for MockEmbeddingModel {
        fn embed(&self, _text: &str) -> Result<EmbeddingOutput> {
            unimplemented!()
        }
        fn model_id(&self) -> String {
            "mock-model".to_string()
        }
        fn embedding_size(&self) -> usize {
            384
        }
    }
}
