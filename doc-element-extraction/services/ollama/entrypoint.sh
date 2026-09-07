#!/bin/bash
# Start Ollama server in the background
ollama serve &

# Wait for server to be ready
echo "Waiting for Ollama server to start..."
for i in $(seq 1 30); do
    if ollama list >/dev/null 2>&1; then
        echo "Ollama server is ready."
        break
    fi
    sleep 2
done

# Pull the default model if not already present
MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
echo "Ensuring model '$MODEL' is available..."
ollama pull "$MODEL" 2>/dev/null || echo "Model pull attempted (may already exist or download in background)."

# Keep the container running
wait
