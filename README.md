# 🚀 FastLLM

A Rust-based LLM inference system that provides OpenAI-compatible API endpoints
for chat completion, with support for loading models directly from HuggingFace.

<img width="1383" alt="image" src="https://github.com/user-attachments/assets/69661c72-0d22-46c4-bf03-1003e94b5b3d" />

<img width="1392" alt="image" src="https://github.com/user-attachments/assets/11995b77-575e-423e-b6e7-c02f4280d1aa" />

# 🧪 Experimental

This is a work in progress and the API is not yet stable!

## ✨ Features

- 🔄 OpenAI-compatible chat completion API endpoint
- 🧮 Text embeddings API endpoint
- 🤗 Load models from HuggingFace Hub (sentence-transformers, qwen, etc.)
- 🍎 MPS Metal, CUDA, and CPU (Auto-detect)

## Model Support

- [x] TinyLlama/TinyLlama-1.1B-Chat-v1.0
- [x] Mistral7 and derivatives
- [x] Qwen2.5 and derivatives
- [x] all-MiniLM-L6-v2 (embeddings)

## 📋 TODO

- [ ] Add support for more models (deepseek, etc.)
- [ ] Add support for more data types (float16, float32, bfloat16, int8, uint8, etc.)
- [ ] Provide benchmarks for different models and configurations compared to other LLM servers
- [ ] Add support for model listing at /v1/models

## 🛠️ Prerequisites

- Rust toolchain (install from https://rustup.rs)
- A HuggingFace account and access token (if you need to access gated models)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fastllm.git
cd fastllm
```

2. Set up your HuggingFace token (if you need to access gated models):
```bash
export HF_TOKEN="your_token_here"
```

3. Build the project:
```bash
cargo build --release
```

## ⚙️ Configuration

Copy the example configuration file and modify it according to your needs:

```bash
cp config.example.json config.json
```

### Configuration Options

```json
{
    "server": {
        "host": "127.0.0.1",  // Server host address
        "port": 3000          // Server port
    },
    "model": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  // HuggingFace model ID
        "revision": "main"                                  // Model revision
    }
}
```

### Environment Variables

You can also configure the server using environment variables with the `FASTLLM_` prefix:

```bash
export FASTLLM_SERVER__HOST=0.0.0.0
export FASTLLM_SERVER__PORT=8080
export FASTLLM_MODEL__MODEL_ID=your-model-id
```

## 🚀 Usage

1. Start the server:
```bash
# Using default config file (config.json)
./target/release/fastllm

# Using a specific config file
./target/release/fastllm --config custom-config.json

# Override the model directly from command line
./target/release/fastllm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The server can be configured in three ways (in order of precedence):
1. Command line arguments (--model)
2. Environment variables (FASTLLM_*)
3. Configuration file (config.json)

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
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## 🔌 API Endpoints

### POST /v1/chat/completions

Create a chat completion. Compatible with OpenAI's chat completion API.

#### Request Format
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

#### Response Format
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

### POST /v1/embeddings

Create embeddings for the input text. Compatible with OpenAI's embeddings API.

#### Request Format
```json
{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "The food was delicious and the waiter..."
}
```

#### Response Format
```json
{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "object": "embedding",
    "embedding": [0.0023, -0.009, 0.015, ...],
    "usage": {
        "prompt_tokens": 57,
        "total_tokens": 57
    }
}
```

## 💬 Chat Message Format

The system uses a simple format to combine messages into a prompt:
- Messages are joined with newlines
- Each message is formatted as `role: content`
- Supported roles: `system`, `user`, `assistant`
- System messages help set the context for the conversation
- Multiple messages can be sent to maintain conversation history

Example prompt format:
```
system: You are a helpful assistant.
user: What is 2+2?
assistant: The answer is 4.
user: Why is that correct?
```

## 📄 License

MIT License
