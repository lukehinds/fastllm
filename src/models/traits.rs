use anyhow::Result;
use candle_core::Tensor;
use crate::models::cache::ModelCache;

pub trait ModelForward {
    fn forward_pass(&self, input: &Tensor, cache: &mut dyn ModelCache) -> Result<Tensor>;
    fn clear_cache(&self);
}

#[allow(unused)]
pub trait ModelGeneration {
    fn sample_next_token(&self, logits: &Tensor, temperature: f32) -> Result<usize>;
    fn is_eos_token(&self, token_id: usize) -> bool;
    fn get_eos_token_id(&self) -> Option<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    struct MockModel {
        cache_cleared: std::cell::Cell<bool>,
    }

    impl ModelForward for MockModel {
        fn forward_pass(&self, _input: &Tensor, _cache: &mut dyn ModelCache) -> Result<Tensor> {
            unimplemented!()
        }

        fn clear_cache(&self) {
            self.cache_cleared.set(true);
        }
    }

    impl ModelGeneration for MockModel {
        fn sample_next_token(&self, _logits: &Tensor, _temperature: f32) -> Result<usize> {
            Ok(42)
        }

        fn is_eos_token(&self, token_id: usize) -> bool {
            token_id == 2
        }

        fn get_eos_token_id(&self) -> Option<usize> {
            Some(2)
        }
    }

    #[test]
    fn test_mock_model_clear_cache() {
        let model = MockModel {
            cache_cleared: std::cell::Cell::new(false),
        };
        
        model.clear_cache();
        assert!(model.cache_cleared.get(), "Cache should be marked as cleared");
    }

    #[test]
    fn test_mock_model_eos_token() {
        let model = MockModel {
            cache_cleared: std::cell::Cell::new(false),
        };

        assert!(model.is_eos_token(2), "Token 2 should be EOS token");
        assert!(!model.is_eos_token(1), "Token 1 should not be EOS token");
        assert_eq!(model.get_eos_token_id(), Some(2));
    }
} 