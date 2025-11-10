# Installing Local LLM for Route Generation

This application supports standalone local LLM models for route generation, ensuring privacy and no API costs.

## Option 1: Ollama (Recommended - Easiest)

Ollama is the easiest way to run local LLMs. It handles model downloads and management automatically.

### Installation

1. **Install Ollama**:
   - Visit https://ollama.ai/
   - Download and install for your platform (Windows/macOS/Linux)
   - Ollama will start automatically as a service

2. **Download a Model**:
   ```bash
   # Recommended: Llama 3.1 8B (good balance of quality and speed)
   ollama pull llama3.1:8b
   
   # Alternative: Mistral 7B (also good for structured output)
   ollama pull mistral:7b
   
   # Smaller option: Phi-3 Mini (faster, less memory)
   ollama pull phi3:mini
   ```

3. **Verify Installation**:
   ```bash
   ollama list
   ```

### Usage

The application will automatically use Ollama if it's running. No additional configuration needed!

**Default Model**: `llama3.1:8b`

## Option 2: Transformers (Fully Embedded)

For a completely standalone solution without external services, use Hugging Face Transformers.

### Installation

1. **Install Dependencies**:
   ```bash
   pip install transformers torch accelerate
   ```

2. **Recommended Models**:
   - **Microsoft Phi-3 Mini** (default): Small, efficient, good for structured output
     - Size: ~2.4 GB
     - RAM: ~4 GB
   - **Llama 3.1 8B**: Better quality, requires more resources
     - Size: ~4.7 GB
     - RAM: ~8 GB
   - **Mistral 7B**: Good balance
     - Size: ~4.1 GB
     - RAM: ~8 GB

### Usage

The application will automatically download and cache the model on first use.

**Note**: First run will download the model (can be 2-8 GB depending on model).

## Comparison

| Feature | Ollama | Transformers |
|---------|--------|--------------|
| **Ease of Setup** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐ Moderate |
| **Memory Usage** | Lower (optimized) | Higher (full model) |
| **Speed** | Fast | Moderate |
| **Model Management** | Automatic | Manual |
| **Standalone** | Requires Ollama service | Fully embedded |
| **Recommended For** | Most users | Advanced users |

## Troubleshooting

### Ollama Not Found

If you see "Ollama server not found":
1. Make sure Ollama is installed and running
2. Check if Ollama is accessible: `curl http://localhost:11434/api/tags`
3. If using custom port, configure in application settings

### Transformers Out of Memory

If you get out-of-memory errors:
1. Use a smaller model (Phi-3 Mini instead of Llama 8B)
2. Enable quantization: `pip install bitsandbytes`
3. Use CPU instead of GPU (slower but uses less memory)

### Model Download Issues

If model download fails:
1. Check internet connection
2. Increase timeout in settings
3. Download model manually from Hugging Face

## Recommended Setup

**For Most Users**: Use Ollama with `llama3.1:8b`
- Easy to set up
- Good quality
- Reasonable speed
- Automatic updates

**For Advanced Users**: Use Transformers with `microsoft/Phi-3-mini-4k-instruct`
- Fully embedded
- No external dependencies
- Smaller memory footprint

