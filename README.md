# FastLLM

> A Rust inference server providing OpenAI-compatible APIs for local LLM deployment. Run language models directly from HuggingFace with native MacOS Metal, CUDA and CPU support.

<div align="center">

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI%20Compatible-412991.svg?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Enabled-yellow.svg?style=for-the-badge)](https://huggingface.co/)

</div>

## Experimental

This is a work in progress and the API is not yet stable!

## Key Features
- Native acceleration on Metal (Apple Silicon) and CUDA
- Direct model loading from HuggingFace Hub
- Run various architectures like Mistral, Qwen, TinyLlama
- Generate embeddings using models like all-MiniLM-L6-v2

## Design Principles

FastLLM adheres to the following core design principles:

**Simple and Modular**
   - Clean, well-documented code structure
   - Modular architecture for easy model integration
   - Trait-based design for flexible model implementations
   - Automatic architecture detection from model configs

**Zero Config**
   - Sensible defaults for all features and optimizations
   - Automatic hardware detection and optimization
   - Smart fallbacks when optimal settings aren't available

**Easy to Extend**
   - Clear separation of concerns
   - Minimal boilerplate for adding new models
   - Comprehensive test coverage and examples
   - Detailed documentation for model integration

The goal is to make it as straightforward as possible to add new models while maintaining high performance by default.

## Supported Models

| Model Family | Supported Architectures | Example Models |
|------------|-----------------|-------------|
| **Llama** | LlamaForCausalLM | • TinyLlama-1.1B-Chat<br>• Any Llama2 derivative |
| **Mistral** | MistralForCausalLM | • Mistral-7B and derivatives<br>• Mixtral-8x7B |
| **Qwen** | Qwen2ForCausalLM<br>Qwen2_5_VLForConditionalGeneration | • Qwen2<br>• Qwen2.5 |
| **BERT** | BertModel<br>RobertaModel<br>DebertaModel | • all-MiniLM-L6-v2<br>• Any BERT/RoBERTa/DeBERTa model |

## Quick Start

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

## Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests
- Share benchmarks and performance reports

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---
