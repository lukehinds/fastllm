# 🚀 FastLLM

> A Rust inference server providing OpenAI-compatible APIs for local LLM deployment. Run language models directly from HuggingFace with native MacOS Metal, CUDA and CPU support.

<div align="center">

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI%20Compatible-412991.svg?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Enabled-yellow.svg?style=for-the-badge)](https://huggingface.co/)

</div>

## 🧪 Experimental

This is a work in progress and the API is not yet stable!

## 🌟 Key Features

- 🚄 **High Performance** - Native acceleration on Metal (Apple Silicon) and CUDA
- 🤗 **HuggingFace Integration** - Direct model loading from HuggingFace Hub
- 🔌 **Multiple Model Support** - Run various architectures like Mistral, Qwen, TinyLlama
- 📊 **Text Embeddings** - Generate embeddings using models like all-MiniLM-L6-v2

## 🎯 Supported Models

| Model Type | Supported Models | Description |
|------------|-----------------|-------------|
| **Chat Models** | • Mistral-7B and derivatives<br>• Qwen2.5 and derivatives<br>• TinyLlama-1.1B-Chat | High-quality instruction-following models |
| **Embedding Models** | • all-MiniLM-L6-v2 | Efficient text embedding generation |

## 🚀 Quick Start

### Prerequisites

- Rust toolchain ([install from rustup.rs](https://rustup.rs))
- HuggingFace token (for gated models)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fastllm.git
cd fastllm

# Optional: Set HuggingFace token for gated models
export HF_TOKEN="your_token_here"

# Build the project (MacOS Metal)
cargo build --release --features "metal"

# Build the project (Linux CUDA)
cargo build --release --features "cuda"

# Build the project (CPU)
cargo build --release
```

### Running the Server

```bash
# Start with default settings
./target/release/fastllm

# Or specify a model directly
./target/release/fastllm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## 🔧 Configuration

FastLLM can be configured through multiple methods (in order of precedence):

1. **Command Line Arguments**
   ```bash
   ./target/release/fastllm --model mistralai/Mistral-7B-v0.1
   ```

2. **Environment Variables**
   ```bash
   export FASTLLM_SERVER__HOST=0.0.0.0
   export FASTLLM_SERVER__PORT=8080
   export FASTLLM_MODEL__MODEL_ID=your-model-id
   ```

3. **Configuration File** (`config.json`)
   ```json
   {
       "server": {
           "host": "127.0.0.1",
           "port": 3000
       },
       "model": {
           "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
           "revision": "main"
       }
   }
   ```

## 🔌 API Examples

### Chat Completion

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "stream": true
  }'
```

### Text Embeddings

```bash
curl http://localhost:3000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "The food was delicious and the service was excellent."
  }'
```

## 🗺️ Roadmap

- [ ] Support for more architectures (DeepSeek, Phi, etc.)
- [ ] Comprehensive benchmarking suite
- [ ] Model management API (/v1/models)
- [ ] Improved caching and optimization
- [ ] Multi-GPU Inference

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests
- Share benchmarks and performance reports

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ by the Stacklok Team
</div>
