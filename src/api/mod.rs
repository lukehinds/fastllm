use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::models::{Model, llama::LlamaWithConfig};

mod chat;

pub fn routes(model: Arc<Mutex<Model<LlamaWithConfig>>>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat::create_chat_completion))
        .route("/v1/models", get(chat::list_models))
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
