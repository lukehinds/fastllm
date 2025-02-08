use axum::{
    extract::{Json, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::models::{embeddings::EmbeddingOutput, ModelWrapper};

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    model: String,
    input: String,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    model: String,
    object: String,
    embedding: Vec<f32>,
    dimensions: usize,
    usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct SimilarityRequest {
    model: String,
    text1: String,
    text2: String,
}

#[derive(Debug, Serialize)]
pub struct SimilarityResponse {
    model: String,
    similarity: f32,
    text1: String,
    text2: String,
}

pub async fn create_embedding(
    State(model): State<Arc<Mutex<ModelWrapper>>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<super::ErrorResponse>)> {
    // Get model lock
    tracing::info!("Handling create_embedding request");
    let model_lock = model.lock().await;

    // Validate model
    let loaded_model_id = model_lock.model_id();
    if request.model != loaded_model_id {
        tracing::error!("Model mismatch: requested model '{}' does not match loaded model '{}'", request.model, loaded_model_id);
        return Err((
            StatusCode::BAD_REQUEST,
            Json(super::ErrorResponse::new(
                format!(
                    "Requested model '{}' does not match loaded model '{}'",
                    request.model, loaded_model_id
                ),
                "model_mismatch",
            )),
        ));
    }

    // Generate embeddings
    let output = model_lock.embed(&request.input).map_err(|e| {
        tracing::error!("Embedding error: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(super::ErrorResponse::new(
                format!("Failed to generate embeddings: {}", e),
                "model_error",
            )),
        )
    })?;

    let EmbeddingOutput {
        embeddings,
        model,
        token_count,
    } = output;

    let dimensions = model_lock.embedding_size();

    let response = EmbeddingResponse {
        model,
        object: "embedding".to_string(),
        embedding: embeddings,
        dimensions,
        usage: Usage {
            prompt_tokens: token_count,
            total_tokens: token_count,
        },
    };

    Ok(Json(response))
}

pub async fn compute_similarity(
    State(model): State<Arc<Mutex<ModelWrapper>>>,
    Json(request): Json<SimilarityRequest>,
) -> Result<Json<SimilarityResponse>, (StatusCode, Json<super::ErrorResponse>)> {
    // Get model lock
    let model_lock = model.lock().await;

    // Validate model
    let loaded_model_id = model_lock.model_id();
    if request.model != loaded_model_id {
        tracing::error!("Model mismatch: requested model '{}' does not match loaded model '{}'", request.model, loaded_model_id);
        return Err((
            StatusCode::BAD_REQUEST,
            Json(super::ErrorResponse::new(
                format!(
                    "Requested model '{}' does not match loaded model '{}'",
                    request.model, loaded_model_id
                ),
                "model_mismatch",
            )),
        ));
    }

    // Compute similarity if we have an embedding model
    if let ModelWrapper::Embedding(model) = &*model_lock {
        if let Ok(similarity) = model.compute_similarity(&request.text1, &request.text2) {
            return Ok(Json(SimilarityResponse {
                model: request.model,
                similarity,
                text1: request.text1,
                text2: request.text2,
            }));
        }
    }
    tracing::error!("Failed to compute similarity or model does not support similarity computation");
    Err((
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(super::ErrorResponse::new(
            "Failed to compute similarity or model does not support similarity computation",
            "model_error",
        )),
    ))
}
