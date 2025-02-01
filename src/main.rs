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
    #[arg(short, long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let _args = Args::parse();

    // Load configuration
    let config = config::Config::load()?;

    // Initialize the model
    let device = candle_core::Device::Cpu;  // TODO: Support GPU
    let dtype = model::parse_dtype(&config.model.dtype)?;
    
    let model = model::load_model(
        &config.model.model_id,
        &config.model.revision,
        dtype,
        &device,
    ).await?;

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
