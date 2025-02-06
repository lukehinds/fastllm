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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_head_dimensions() {
        let config = BaseModelConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            vocab_size: 32000,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            num_key_value_heads: Some(8),
            rms_norm_eps: 1e-6,
            rope_theta: Some(10000.0),
            max_position_embeddings: Some(2048),
            sliding_window: Some(512),
            torch_dtype: None,
        };

        let head_dim = config.validate_head_dimensions().unwrap();
        assert_eq!(head_dim, 64, "Head dimension should be 64");
    }

    #[test]
    fn test_invalid_head_dimensions() {
        let config = BaseModelConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            vocab_size: 32000,
            num_hidden_layers: 8,
            num_attention_heads: 7, // Not divisible evenly
            num_key_value_heads: Some(7),
            rms_norm_eps: 1e-6,
            rope_theta: Some(10000.0),
            max_position_embeddings: Some(2048),
            sliding_window: Some(512),
            torch_dtype: None,
        };

        assert!(config.validate_head_dimensions().is_err(), 
            "Should error when hidden_size not divisible by num_attention_heads");
    }

    #[test]
    fn test_valid_gqa_config() {
        let config = BaseModelConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            vocab_size: 32000,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            num_key_value_heads: Some(4), // 8 is divisible by 4
            rms_norm_eps: 1e-6,
            rope_theta: Some(10000.0),
            max_position_embeddings: Some(2048),
            sliding_window: Some(512),
            torch_dtype: None,
        };

        assert!(config.validate_gqa_config().is_ok(), 
            "Valid GQA config should pass validation");
    }

    #[test]
    fn test_invalid_gqa_config() {
        let config = BaseModelConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            vocab_size: 32000,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            num_key_value_heads: Some(3), // 8 is not divisible by 3
            rms_norm_eps: 1e-6,
            rope_theta: Some(10000.0),
            max_position_embeddings: Some(2048),
            sliding_window: Some(512),
            torch_dtype: None,
        };

        assert!(config.validate_gqa_config().is_err(),
            "Should error when num_attention_heads not divisible by num_key_value_heads");
    }
} 