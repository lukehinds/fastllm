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