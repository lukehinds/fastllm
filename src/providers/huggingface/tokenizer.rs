use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

pub fn load_tokenizer(model_id: &str, tokenizer_path: &Path) -> Result<Tokenizer> {
    // For Qwen models, try to load from tokenizer.model if tokenizer.json fails
    if model_id.contains("qwen") {
        let model_path = tokenizer_path.with_file_name("tokenizer.model");
        if model_path.exists() {
            return Tokenizer::from_file(&model_path).map_err(|e| {
                anyhow::anyhow!("Failed to load Qwen tokenizer from {:?}: {}", model_path, e)
            });
        }
    }

    // For Mistral models, try to load with specific options
    if model_id.contains("mistral") {
        // Try loading with specific options for Mistral
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load Mistral tokenizer from {:?}: {}",
                tokenizer_path,
                e
            )
        })?;

        // TODO: Configure Mistral-specific settings if needed
        return Ok(tokenizer);
    }

    // Default loading for other models
    Tokenizer::from_file(tokenizer_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load tokenizer for model '{}' from {:?}: {}",
            model_id,
            tokenizer_path,
            e
        )
    })
}
