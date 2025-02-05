use anyhow::Result;
use serde::Deserialize;
use candle_nn::Activation;

#[derive(Deserialize, Clone, Debug)]
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
}

pub trait ModelConfigValidation {
    fn validate_head_dimensions(&self) -> Result<usize>;
    fn validate_gqa_config(&self) -> Result<()>;
    fn get_activation() -> Activation where Self: Sized {
        Activation::Silu
    }
}

impl ModelConfigValidation for BaseModelConfig {
    fn validate_head_dimensions(&self) -> Result<usize> {
        let head_dim = self.hidden_size / self.num_attention_heads;
        anyhow::ensure!(
            head_dim * self.num_attention_heads == self.hidden_size,
            "hidden_size must be divisible by num_attention_heads"
        );
        anyhow::ensure!(
            head_dim % 2 == 0,
            "head_dim must be even for RoPE embeddings"
        );
        Ok(head_dim)
    }
    
    fn validate_gqa_config(&self) -> Result<()> {
        if let Some(num_kv_heads) = self.num_key_value_heads {
            anyhow::ensure!(
                self.num_attention_heads % num_kv_heads == 0,
                "num_attention_heads must be divisible by num_key_value_heads"
            );
        }
        Ok(())
    }
} 