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