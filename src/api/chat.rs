use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{sse::{Event, Sse}, IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use futures::stream::{self, StreamExt};
use std::convert::Infallible;

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
    #[serde(default)]
    stream: bool,
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

#[derive(Debug, Serialize)]
pub struct ChatCompletionStreamResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatCompletionStreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionStreamChoice {
    index: usize,
    delta: ChatCompletionStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionStreamDelta {
    role: Option<String>,
    content: Option<String>,
}

fn default_max_tokens() -> usize {
    256
}

pub async fn create_chat_completion(
    State(model): State<Arc<Mutex<ModelWrapper>>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<super::ErrorResponse>)> {
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
    
    if request.stream {
        let stream_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let model_id = request.model.clone();

        // Send initial response with role
        let initial_response = ChatCompletionStreamResponse {
            id: stream_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id.clone(),
            choices: vec![ChatCompletionStreamChoice {
                index: 0,
                delta: ChatCompletionStreamDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };

        let mut model = model.lock().await;
        let token_stream = model.generate_stream(&prompt, request.max_tokens, request.temperature)
            .map_err(|e| {
                tracing::error!("Stream generation error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(super::ErrorResponse::new(
                        format!("Failed to generate stream: {}", e),
                        "model_error",
                    )),
                )
            })?;

        let stream_id_final = stream_id.clone();
        let model_id_final = model_id.clone();
        
        let stream = stream::once(futures::future::ok::<_, Infallible>(Event::default().data(serde_json::to_string(&initial_response).unwrap())))
            .chain(token_stream.map(move |token_result| {
                let token = token_result.unwrap_or_else(|e| {
                    tracing::error!("Token generation error: {}", e);
                    String::new()
                });

                let stream_response = ChatCompletionStreamResponse {
                    id: stream_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id.clone(),
                    choices: vec![ChatCompletionStreamChoice {
                        index: 0,
                        delta: ChatCompletionStreamDelta {
                            role: None,
                            content: Some(token),
                        },
                        finish_reason: None,
                    }],
                };

                Ok::<_, Infallible>(Event::default().data(serde_json::to_string(&stream_response).unwrap()))
            }))
            .chain(stream::once(futures::future::ok::<_, Infallible>({
                let final_response = ChatCompletionStreamResponse {
                    id: stream_id_final,
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id_final,
                    choices: vec![ChatCompletionStreamChoice {
                        index: 0,
                        delta: ChatCompletionStreamDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                };

                Event::default().data(serde_json::to_string(&final_response).unwrap())
            })));

        Ok(Sse::new(stream).into_response())
    } else {
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

        Ok(Json(response).into_response())
    }
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
