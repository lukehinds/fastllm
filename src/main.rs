use anyhow::Result;
use axum::Router;
use clap::Parser;
use std::{sync::Arc, net::SocketAddr};
use tokio::sync::Mutex;
use tower_http::trace::TraceLayer;

mod api;
mod config;
mod model;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to config file
    #[arg(short, long, default_value = "config.json")]
    config: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Parse command line arguments
    let args = Args::parse();
    tracing::info!("Loading config from: {}", args.config);

    // Load configuration
    let config = config::Config::from_file(&args.config)?;
    tracing::info!("Config loaded: {:?}", config);

    // Initialize the model
    let device = candle_core::Device::new_metal(0)?;
    // let device = candle_core::Device::Cpu;  // Use CPU device
    let dtype = model::parse_dtype(&config.model.dtype)?;
    tracing::info!("Initializing model with dtype: {:?} on device: {:?}", dtype, device);
    
    tracing::info!("Loading model: {}", config.model.model_id);
    let model = model::load_model(
        &config.model.model_id,
        &config.model.revision,
        dtype,
        &device,
    ).await?;
    tracing::info!("Model loaded successfully");

    // Create shared model state
    let model = Arc::new(Mutex::new(model));

    // Build the API router
    let app = Router::new()
        .merge(api::routes(model.clone()))
        .layer(TraceLayer::new_for_http());

    // Start the server
    let addr = format!("{}:{}", config.server.host, config.server.port)
        .parse::<SocketAddr>()
        .expect("Invalid address");

    tracing::info!("Starting server on {}", addr);
    axum::serve(
        tokio::net::TcpListener::bind(&addr).await?,
        app.into_make_service(),
    )
    .await?;

    Ok(())
}
