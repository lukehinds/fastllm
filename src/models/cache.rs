use std::fmt::Debug;

#[allow(unused)]
pub trait ModelCache: Send + Debug {
    fn increment_offset(&mut self);
    fn reset(&mut self);
    fn get_offset(&self) -> usize;
}

#[derive(Debug)]
pub struct CommonCache {
    seqlen_offset: usize,
}

impl CommonCache {
    pub fn new() -> Self {
        Self { seqlen_offset: 0 }
    }
}

impl ModelCache for CommonCache {
    fn increment_offset(&mut self) {
        self.seqlen_offset += 1;
        tracing::debug!("Cache seqlen_offset incremented to {}", self.seqlen_offset);
    }
    
    fn reset(&mut self) {
        self.seqlen_offset = 0;
        tracing::debug!("Cache reset");
    }
    
    fn get_offset(&self) -> usize {
        self.seqlen_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_cache_operations() {
        let mut cache = CommonCache::new();
        assert_eq!(cache.get_offset(), 0, "Initial offset should be 0");

        cache.increment_offset();
        assert_eq!(cache.get_offset(), 1, "Offset should be 1 after increment");

        cache.increment_offset();
        assert_eq!(cache.get_offset(), 2, "Offset should be 2 after second increment");

        cache.reset();
        assert_eq!(cache.get_offset(), 0, "Offset should be 0 after reset");
    }
} 