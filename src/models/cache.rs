use std::fmt::Debug;
use std::any::Any;

#[allow(unused)]
pub trait ModelCache: Send + Debug {
    fn increment_offset(&mut self);
    fn reset(&mut self);
    fn get_offset(&self) -> usize;
    fn as_any_mut(&mut self) -> &mut dyn Any;
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

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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

    #[test]
    fn test_common_cache_as_any() {
        let mut cache = CommonCache::new();
        let any_cache = cache.as_any_mut();
        assert!(any_cache.downcast_mut::<CommonCache>().is_some(), "Should be able to downcast to CommonCache");
        assert!(any_cache.downcast_mut::<String>().is_none(), "Should not be able to downcast to wrong type");
    }

    #[test]
    fn test_common_cache_debug() {
        let cache = CommonCache::new();
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("CommonCache"), "Debug output should contain type name");
    }
} 