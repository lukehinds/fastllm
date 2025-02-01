# infrs

A Rust-based LLM inference system that provides OpenAI-compatible API endpoints for chat completion, with support for loading models from HuggingFace.

## Features

- OpenAI-compatible chat completion API endpoint
- Support for loading models from HuggingFace Hub
- Configurable server settings
- CPU-based inference with efficient tensor operations using Candle
- Support for multiple data types (float16, float32, bfloat16)

## Prerequisites

- Rust toolchain (install from https://rustup.rs)
- A HuggingFace account and access token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/infrs.git
cd infrs
```

2. Set up your HuggingFace token (if you need to access gated models):
```bash
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

3. Build the project:
```bash
cargo build --release
```

## Configuration

Copy the example configuration file and modify it according to your needs:

```bash
cp config.example.json config.json
```

Configuration options:

```json
{
    "server": {
        "host": "127.0.0.1",  // Server host address
        "port": 3000          // Server port
    },
    "model": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  // HuggingFace model ID
        "revision": "main",                                // Model revision
        "dtype": "float32"                                // Data type (float16, float32, bfloat16)
    }
}
```

You can also configure the server using environment variables with the `INFRS_` prefix:

```bash
export INFRS_SERVER__HOST=0.0.0.0
export INFRS_SERVER__PORT=8080
export INFRS_MODEL__MODEL_ID=your-model-id
```

## Usage

1. Start the server:
```bash
./target/release/infrs
```

2. Make requests to the OpenAI-compatible endpoints:

```bash
# Chat completion
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is 2+2?"
      }
    ],
    "max_tokens": 50
  }'

# List available models
curl http://localhost:3000/v1/models
```

## API Endpoints

### POST /v1/chat/completions

Create a chat completion. Compatible with OpenAI's chat completion API.

Request body:
```json
{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello! How are you?"
        }
    ],
    "max_tokens": 256,
    "temperature": 0.7
}
```

Response format:
```json
{
    "id": "chatcmpl-123abc...",
    "object": "chat.completion",
    "created": 1707123456,
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 4."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 57,
        "completion_tokens": 23,
        "total_tokens": 80
    }
}
```

### GET /v1/models

List available models. Returns the configured model with its details.

## Recommended Models for Testing

For initial testing, we recommend using smaller models to ensure everything is working correctly:

1. TinyLlama (1.1B parameters):
```json
{
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "revision": "main",
    "dtype": "float32"
}
```

2. For larger models like Llama-2, ensure you have sufficient RAM and consider using float16:
```json
{
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
    "revision": "main",
    "dtype": "float16"
}
```

## Chat Message Format

The system uses a simple format to combine messages into a prompt:
- Messages are joined with newlines
- Each message is formatted as "role: content"
- Supported roles: "system", "user", "assistant"
- System messages help set the context for the conversation
- Multiple messages can be sent to maintain conversation history

Example prompt format:
```
system: You are a helpful assistant.
user: What is 2+2?
assistant: The answer is 4.
user: Why is that correct?
```

## Troubleshooting

1. Permission Issues:
   - Ensure your HuggingFace token is set correctly
   - For gated models (like Llama-2), make sure you have accepted the model's license on HuggingFace

2. Memory Issues:
   - Start with a smaller model like TinyLlama
   - Try using "float16" dtype instead of "float32"
   - Ensure you have enough available RAM for your chosen model

3. Performance:
   - The system currently uses CPU-only inference
   - Large models will be slower; consider using smaller models for testing
   - Response generation speed depends on the model size and available CPU resources

4. API Usage:
   - Make sure to use the exact model ID in both config.json and API requests
   - Include a system message to help guide the model's behavior
   - Keep max_tokens reasonable (50-256 for most use cases)

## License

MIT License
