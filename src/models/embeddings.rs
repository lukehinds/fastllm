use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, Module, embedding, linear};
use serde::Deserialize;
use tokenizers::Tokenizer;
use crate::providers::huggingface::load_tokenizer;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

#[derive(Debug)]
pub struct EmbeddingOutput {
    pub embeddings: Vec<f32>,
    pub model: String,
    pub token_count: usize,
}

pub trait EmbeddingModel {
    fn embed(&self, text: &str) -> Result<EmbeddingOutput>;
    fn model_id(&self) -> String;
    fn embedding_size(&self) -> usize;
    
    fn compute_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        // Get embeddings for both texts
        let emb1 = self.embed(text1)?;
        let emb2 = self.embed(text2)?;

        // Convert to vectors
        let v1 = &emb1.embeddings;
        let v2 = &emb2.embeddings;

        // Compute cosine similarity
        let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        Ok(dot_product / (norm1 * norm2))
    }
}

#[derive(Deserialize, Clone)]
pub struct SentenceBertConfig {
    pub max_seq_length: usize,
    pub do_lower_case: bool,
}

#[derive(Deserialize, Clone)]
pub struct BertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
}

pub struct BertLayer {
    attention: BertAttention,
    intermediate: candle_nn::Linear,
    output: candle_nn::Linear,
    layer_norm2: candle_nn::LayerNorm,
}

pub struct BertAttention {
    query: candle_nn::Linear,
    key: candle_nn::Linear,
    value: candle_nn::Linear,
    output: candle_nn::Linear,
    layer_norm: candle_nn::LayerNorm,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl BertAttention {
    fn new(vb: VarBuilder, prefix: &str, config: &BertConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = hidden_size / num_attention_heads;
        
        Ok(Self {
            query: linear(hidden_size, hidden_size, vb.pp(format!("{}.attention.self.query", prefix)))
                .map_err(|e| anyhow::anyhow!("Failed to create query layer: {}", e))?,
            key: linear(hidden_size, hidden_size, vb.pp(format!("{}.attention.self.key", prefix)))
                .map_err(|e| anyhow::anyhow!("Failed to create key layer: {}", e))?,
            value: linear(hidden_size, hidden_size, vb.pp(format!("{}.attention.self.value", prefix)))
                .map_err(|e| anyhow::anyhow!("Failed to create value layer: {}", e))?,
            output: linear(hidden_size, hidden_size, vb.pp(format!("{}.attention.output.dense", prefix)))
                .map_err(|e| anyhow::anyhow!("Failed to create attention output layer: {}", e))?,
            layer_norm: candle_nn::layer_norm(hidden_size, config.layer_norm_eps, vb.pp(format!("{}.attention.output.LayerNorm", prefix)))
                .map_err(|e| anyhow::anyhow!("Failed to create attention layer norm: {}", e))?,
            num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_length, _) = x.dims3()
            .map_err(|e| anyhow::anyhow!("Failed to get tensor dimensions: {}", e))?;
        x.reshape((
            batch_size,
            seq_length,
            self.num_attention_heads,
            self.attention_head_size,
        ))
        .map_err(|e| anyhow::anyhow!("Failed to reshape tensor: {}", e))?
        .permute((0, 2, 1, 3))
        .map_err(|e| anyhow::anyhow!("Failed to permute tensor: {}", e))
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let query = self.query.forward(hidden_states)
            .map_err(|e| anyhow::anyhow!("Failed in query forward pass: {}", e))?;
        let key = self.key.forward(hidden_states)
            .map_err(|e| anyhow::anyhow!("Failed in key forward pass: {}", e))?;
        let value = self.value.forward(hidden_states)
            .map_err(|e| anyhow::anyhow!("Failed in value forward pass: {}", e))?;

        let query = self.transpose_for_scores(&query)?;
        let key = self.transpose_for_scores(&key)?;
        let value = self.transpose_for_scores(&value)?;

        let _key_size = key.dim(key.dims().len() - 1)
            .map_err(|e| anyhow::anyhow!("Failed to get key size: {}", e))?;
        let key_transposed = key.transpose(key.dims().len() - 2, key.dims().len() - 1)
            .map_err(|e| anyhow::anyhow!("Failed to transpose key: {}", e))?;

        let attention_scores = query.matmul(&key_transposed)
            .map_err(|e| anyhow::anyhow!("Failed to compute attention scores: {}", e))?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())
            .map_err(|e| anyhow::anyhow!("Failed to scale attention scores: {}", e))?;
        let attention_probs = candle_nn::ops::softmax(&attention_scores, attention_scores.dims().len() - 1)
            .map_err(|e| anyhow::anyhow!("Failed to compute attention probabilities: {}", e))?;

        let context = attention_probs.matmul(&value)
            .map_err(|e| anyhow::anyhow!("Failed to compute context: {}", e))?;
        let context = context.permute((0, 2, 1, 3))
            .map_err(|e| anyhow::anyhow!("Failed to permute context: {}", e))?;
        let (batch_size, seq_length, _, _) = context.dims4()
            .map_err(|e| anyhow::anyhow!("Failed to get context dimensions: {}", e))?;
        let context = context.reshape((batch_size, seq_length, self.num_attention_heads * self.attention_head_size))
            .map_err(|e| anyhow::anyhow!("Failed to reshape context: {}", e))?;

        let attention_output = self.output.forward(&context)
            .map_err(|e| anyhow::anyhow!("Failed in attention output forward pass: {}", e))?;
        let hidden_states = hidden_states.add(&attention_output)
            .map_err(|e| anyhow::anyhow!("Failed to add attention output: {}", e))?;
        self.layer_norm.forward(&hidden_states)
            .map_err(|e| anyhow::anyhow!("Failed in attention layer norm: {}", e))
    }
}

impl BertLayer {
    fn new(vb: VarBuilder, prefix: &str, config: &BertConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        
        Ok(Self {
            attention: BertAttention::new(vb.clone(), prefix, config)?,
            intermediate: linear(hidden_size, intermediate_size, vb.pp(format!("{}.intermediate.dense", prefix)))
                .map_err(|e| anyhow::anyhow!("Failed to create intermediate layer: {}", e))?,
            output: linear(intermediate_size, hidden_size, vb.pp(format!("{}.output.dense", prefix)))
                .map_err(|e| anyhow::anyhow!("Failed to create output layer: {}", e))?,
            layer_norm2: candle_nn::layer_norm(hidden_size, config.layer_norm_eps, vb.pp(format!("{}.output.LayerNorm", prefix)))
                .map_err(|e| anyhow::anyhow!("Failed to create layer norm: {}", e))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.attention.forward(hidden_states)?;
        
        let intermediate_output = self.intermediate.forward(&hidden_states)
            .map_err(|e| anyhow::anyhow!("Failed in intermediate forward pass: {}", e))?;
        let intermediate_output = intermediate_output.gelu()
            .map_err(|e| anyhow::anyhow!("Failed to apply GELU activation: {}", e))?;
        let layer_output = self.output.forward(&intermediate_output)
            .map_err(|e| anyhow::anyhow!("Failed in output forward pass: {}", e))?;
        let hidden_states = hidden_states.add(&layer_output)
            .map_err(|e| anyhow::anyhow!("Failed to add layer output: {}", e))?;
        self.layer_norm2.forward(&hidden_states)
            .map_err(|e| anyhow::anyhow!("Failed in layer norm: {}", e))
    }
}

pub struct MiniLMModel {
    tokenizer: Tokenizer,
    word_embeddings: candle_nn::Embedding,
    position_embeddings: candle_nn::Embedding,
    layer_norm: candle_nn::LayerNorm,
    encoder_layers: Vec<BertLayer>,
    device: Device,
    model_id: String,
    do_lower_case: bool,
}

impl MiniLMModel {
    pub fn new(model_id: &str) -> Result<Self> {
        // Initialize HF API with token if available
        let token = std::env::var("HF_TOKEN").ok();
        let api = ApiBuilder::new()
            .with_token(token)
            .build()?;
        
        // Download tokenizer and model files
        tracing::info!("Downloading files from HuggingFace...");
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));
        
        let tokenizer_path = repo.get("tokenizer.json")?;
        let model_path = repo.get("model.safetensors")?;
        let config_path = repo.get("config.json")?;
        let sentence_config_path = repo.get("sentence_bert_config.json")?;
        
        // Load configs first
        let config_content = std::fs::read_to_string(&config_path)?;
        let config: BertConfig = serde_json::from_str(&config_content)?;
        
        let sentence_config_content = std::fs::read_to_string(&sentence_config_path)?;
        let sentence_config: SentenceBertConfig = serde_json::from_str(&sentence_config_content)?;
        
        // Load tokenizer
        let tokenizer = load_tokenizer(model_id, &tokenizer_path)?;
        
        tracing::info!("Model config: hidden_size={}, max_position_embeddings={}, max_seq_length={}, do_lower_case={}", 
            config.hidden_size, config.max_position_embeddings, sentence_config.max_seq_length, sentence_config.do_lower_case);
        
        // Load model weights
        let device = Device::Cpu; // MiniLM is small enough to run on CPU
        let weights = std::fs::read(model_path)?;
        let tensors = candle_core::safetensors::load_buffer(&weights, &device)
            .map_err(|e| anyhow::anyhow!("Failed to load model weights: {}", e))?;
        
        tracing::info!("Available tensors:");
        for name in tensors.keys() {
            tracing::info!("  {}", name);
        }
        
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        
        // Initialize model components
        let word_embeddings = embedding(
            tokenizer.get_vocab_size(false),
            config.hidden_size,
            vb.pp("embeddings.word_embeddings"),
        ).map_err(|e| anyhow::anyhow!("Failed to create word embeddings: {}", e))?;
        
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("embeddings.position_embeddings"),
        ).map_err(|e| anyhow::anyhow!("Failed to create position embeddings: {}", e))?;
        
        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            1e-12, // eps
            vb.pp("embeddings.LayerNorm"),
        ).map_err(|e| anyhow::anyhow!("Failed to create layer norm: {}", e))?;
        
        // Initialize encoder layers
        let mut encoder_layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = BertLayer::new(
                vb.clone(),
                &format!("encoder.layer.{}", i),
                &config,
            )?;
            encoder_layers.push(layer);
        }
        
        Ok(Self {
            tokenizer,
            word_embeddings,
            position_embeddings,
            layer_norm,
            encoder_layers,
            device,
            model_id: model_id.to_string(),
            do_lower_case: sentence_config.do_lower_case,
        })
    }

    fn normalize_l2(&self, v: &Tensor) -> Result<Tensor> {
        // Normalize using L2 norm (Euclidean norm)
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }

    fn mean_pooling(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (_n_sentence, _n_tokens, hidden_size) = embeddings.dims3()?;
        
        // Expand attention mask to match embedding dimensions
        let attention_mask = attention_mask.unsqueeze(2)?  // Add hidden dimension
            .expand((attention_mask.dim(0)?, attention_mask.dim(1)?, hidden_size))?;
        let attention_mask = attention_mask.to_dtype(embeddings.dtype())?;
        
        // Apply attention mask to embeddings
        let masked_embeddings = embeddings.mul(&attention_mask)?;
        
        // Sum the masked embeddings
        let summed = masked_embeddings.sum(1)?;
        
        // Get the number of actual tokens (sum of attention mask for each sentence)
        let n_tokens = attention_mask.sum(1)?.sum(1)?.unsqueeze(1)?;
        
        // Average by dividing by the number of actual tokens
        let mean_pooled = summed.broadcast_div(&n_tokens)?;
        
        Ok(mean_pooled)
    }

    fn embed_tokens(&self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let word_embeds = self.word_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(position_ids)?;
        
        let embeddings = word_embeds.add(&position_embeds)?;
        self.layer_norm.forward(&embeddings)
            .map_err(|e| anyhow::anyhow!("Failed to apply layer norm: {}", e))
    }

    fn forward(&self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // Get embeddings
        let mut hidden_states = self.embed_tokens(input_ids, position_ids)
            .map_err(|e| anyhow::anyhow!("Failed to get embeddings: {}", e))?;
        
        // Pass through transformer layers
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        
        // Just return the hidden states - we'll do pooling later
        Ok(hidden_states)
    }
}

impl EmbeddingModel for MiniLMModel {
    fn embed(&self, text: &str) -> Result<EmbeddingOutput> {
        // Preprocess text
        let text = if self.do_lower_case {
            text.to_lowercase()
        } else {
            text.to_owned()
        };
        
        // Tokenize input
        let encoding = self.tokenizer.encode(text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let token_count = input_ids.len();
        
        // Create position IDs
        let position_ids: Vec<u32> = (0..token_count).map(|i| i as u32).collect();
        
        // Convert to tensors
        let input_tensor = Tensor::new(input_ids, &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?
            .unsqueeze(0)?;
        let position_tensor = Tensor::new(&position_ids[..], &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create position tensor: {}", e))?
            .unsqueeze(0)?;
        let attention_tensor = Tensor::new(attention_mask, &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create attention tensor: {}", e))?
            .unsqueeze(0)?;
        
        // Forward pass through transformer
        let embeddings = self.forward(&input_tensor, &position_tensor)?;
        
        // Mean pooling
        let pooled = self.mean_pooling(&embeddings, &attention_tensor)?;
        
        // L2 normalize
        let normalized = self.normalize_l2(&pooled)?;
        
        // Remove batch dimension and convert to vector
        let normalized = normalized.squeeze(0)?;  // Remove batch dimension
        let embeddings = normalized.to_vec1()?;
        
        Ok(EmbeddingOutput {
            embeddings,
            model: self.model_id(),
            token_count,
        })
    }

    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn embedding_size(&self) -> usize {
        384 // MiniLM-L6-v2 embedding size
    }
}

// Add test module
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_similarities() -> Result<()> {
        let model = MiniLMModel::new("sentence-transformers/all-MiniLM-L6-v2")?;
        
        // Test similar sentences (paraphrases)
        let sim1 = model.compute_similarity(
            "I really enjoyed the movie. It was a great film.",
            "The movie was excellent and I had a good time watching it."
        )?;
        assert!(
            sim1 > 0.8, 
            "Similar sentences should have high similarity (got {}, expected > 0.8)", 
            sim1
        );
        
        // Test somewhat related sentences (same topic, different aspects)
        let sim2 = model.compute_similarity(
            "I enjoy programming in Python because it's easy to read.",
            "Java is a popular programming language for enterprise applications."
        )?;
        assert!(
            sim2 > 0.4 && sim2 < 0.8,
            "Somewhat related sentences should have moderate similarity (got {}, expected between 0.4 and 0.8)",
            sim2
        );
        
        // Test completely dissimilar sentences (different topics)
        let sim3 = model.compute_similarity(
            "The recipe calls for two cups of flour and one cup of sugar.",
            "The Hubble telescope has captured stunning images of distant galaxies."
        )?;
        assert!(
            sim3 < 0.4,
            "Dissimilar sentences should have low similarity (got {}, expected < 0.4)",
            sim3
        );
        
        Ok(())
    }
} 