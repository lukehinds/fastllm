use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

pub mod llama;
pub mod mistral;
pub mod qwen;
pub mod model_initializer;
use crate::providers::huggingface;

pub use huggingface::load_model;
use model_initializer::ModelInitializer;

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
            let last_logits = logits.get(0)?;  // Get first batch
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
