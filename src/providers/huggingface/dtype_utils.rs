use candle_core::DType;

pub trait TorchDTypeConverter {
    fn torch_dtype_to_candle(torch_dtype: &str) -> DType;
}

pub struct DefaultDTypeConverter;

impl TorchDTypeConverter for DefaultDTypeConverter {
    fn torch_dtype_to_candle(torch_dtype: &str) -> DType {
        match torch_dtype {
            "float16" => DType::F16,
            "float32" => DType::F32,
            "float64" => DType::F64,
            "bfloat16" => DType::BF16,
            _ => {
                tracing::warn!("Unsupported torch_dtype: {}, defaulting to F32", torch_dtype);
                DType::F32
            }
        }
    }
}

pub fn get_dtype(config_torch_dtype: Option<&String>, default_dtype: DType) -> DType {
    let dtype = config_torch_dtype
        .map(|dt| {
            tracing::info!("Model config specifies torch_dtype: {}", dt);
            let candle_dtype = DefaultDTypeConverter::torch_dtype_to_candle(dt);
            tracing::info!("Converted to candle dtype: {:?}", candle_dtype);
            candle_dtype
        })
        .unwrap_or_else(|| {
            tracing::info!("No torch_dtype specified, using default: {:?}", default_dtype);
            default_dtype
        });
    
    tracing::info!("Final dtype being used: {:?}", dtype);
    dtype
}

pub fn validate_dtype_compatibility(dtype: DType, model_name: &str) {
    tracing::info!("Validating dtype compatibility for model: {}", model_name);
    tracing::info!("Current dtype: {:?}", dtype);
    // Add any model-specific dtype validation logic here
    // For example, if certain models require specific dtypes, you can check and warn here
}
