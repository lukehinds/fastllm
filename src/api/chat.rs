use axum::{
    extract::{Json, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::models::ModelWrapper;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    index: usize,
    message: ChatCompletionMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

fn default_max_tokens() -> usize {
    256
}

pub async fn create_chat_completion(
    State(model): State<Arc<Mutex<ModelWrapper>>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<super::ErrorResponse>)> {
    // Validate that the requested model matches the loaded model
    let model_lock = model.lock().await;
    let loaded_model_id = model_lock.model_id();
    if request.model != loaded_model_id {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(super::ErrorResponse::new(
                format!("Requested model '{}' does not match loaded model '{}'", request.model, loaded_model_id),
                "model_mismatch",
            )),
        ));
    }
    drop(model_lock);  // Release the lock before generation

    // Format the conversation history into a prompt
    let prompt = format_messages(&request.messages);
    tracing::debug!("Formatted prompt: {}", prompt);
    
    // Generate the response
    let mut model = model.lock().await;
    let output = model.generate(&prompt, request.max_tokens, request.temperature)
        .map_err(|e| {
            tracing::error!("Generation error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(super::ErrorResponse::new(
                    format!("Failed to generate response: {}", e),
                    "model_error",
                )),
            )
        })?;

    let output_len = output.len();
    let prompt_len = prompt.len();

    tracing::debug!("Generated response: {}", output);

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: request.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: output,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: output_len,
            total_tokens: prompt_len + output_len,
        },
    };

    Ok(Json(response))
}

pub async fn list_models(
    State(model): State<Arc<Mutex<ModelWrapper>>>,
) -> Json<serde_json::Value> {
    let model = model.lock().await;
    let model_id = model.model_id();
    
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": chrono::Utc::now().timestamp(),
                "owned_by": "local",
            }
        ]
    }))
}

fn format_messages(messages: &[ChatMessage]) -> String {
    let mut formatted = String::new();
    
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                formatted.push_str("<|system|>\n");
                formatted.push_str(&msg.content);
                formatted.push_str("\n</s>\n");
            },
            "user" => {
                formatted.push_str("<|user|>\n");
                formatted.push_str(&msg.content);
                formatted.push_str("\n</s>\n");
            },
            "assistant" => {
                formatted.push_str("<|assistant|>\n");
                formatted.push_str(&msg.content);
                formatted.push_str("\n</s>\n");
            },
            _ => {
                tracing::warn!("Unknown role: {}", msg.role);
                formatted.push_str(&format!("{}: {}\n", msg.role, msg.content));
            }
        }
    }
    
    // Add the final assistant prompt
    formatted.push_str("<|assistant|>\n");
    
    formatted
}
