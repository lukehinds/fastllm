use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

mod huggingface;

pub use huggingface::load_model;

pub struct Model {
    tokenizer: Tokenizer,
    model: candle_transformers::models::llama::Llama,
    device: Device,
    logits_processor: LogitsProcessor,
}

impl Model {
    pub fn new(
        tokenizer: Tokenizer,
        model: candle_transformers::models::llama::Llama,
        device: Device,
    ) -> Self {
        Self {
            tokenizer,
            model,
            device,
            logits_processor: LogitsProcessor::new(Default::default(), None, None),
        }
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = encoding.get_ids();
        
        let input_tensor = Tensor::new(input_ids, &self.device)?;
        let mut output_ids = Vec::new();
        let mut next_token = input_tensor;
        let mut pos = 0;
        
        for _ in 0..max_tokens {
            let logits = self.model.forward(&next_token, pos)?;
            let logits = logits.squeeze(0)?;
            
            let next_token_id = self.logits_processor.sample(&logits)?;
            if let Some(eos_token_id) = self.tokenizer.token_to_id("</s>") {
                if next_token_id == eos_token_id {
                    break;
                }
            }
            
            output_ids.push(next_token_id);
            next_token = Tensor::new(&[next_token_id], &self.device)?;
            pos += 1;
        }
        
        self.tokenizer.decode(&output_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))
    }
}

pub fn parse_dtype(dtype: &str) -> Result<DType> {
    match dtype.to_lowercase().as_str() {
        "float32" | "f32" => Ok(DType::F32),
        "float16" | "f16" => Ok(DType::F16),
        "bfloat16" | "bf16" => Ok(DType::BF16),
        _ => anyhow::bail!("Unsupported dtype: {}", dtype),
    }
}
