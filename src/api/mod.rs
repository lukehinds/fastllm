use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::models::ModelWrapper;

mod chat;
mod embeddings;

pub fn routes(model: Arc<Mutex<ModelWrapper>>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat::create_chat_completion))
        .route("/v1/models", get(chat::list_models))
        .route("/v1/embeddings", post(embeddings::create_embedding))
        .route("/v1/similarity", post(embeddings::compute_similarity))
        .with_state(model)
}

#[derive(Debug, serde::Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, serde::Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

impl ErrorResponse {
    pub fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                r#type: error_type.into(),
                code: None,
            },
        }
    }
}
