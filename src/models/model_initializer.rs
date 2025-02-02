use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

pub trait ModelInitializer {
    type Config;
    type Cache;

    fn initialize_model(
        config: &Self::Config,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<(Self, Self::Cache)>
    where
        Self: Sized;

    fn forward(
        &self,
        input: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor>;
}
