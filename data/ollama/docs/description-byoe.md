## Bring Your Own Endpoint (BYOE)

This service connects to your own Ollama server. You need to:

1. **Install Ollama** on your machine or server: [ollama.com/download](https://ollama.com/download)
2. **Pull the model**: `ollama pull <model-name>`
3. **Ensure your Ollama server is accessible** from the network
4. **Provide the base URL** during enrollment (default: `http://localhost:11434`)

## Best Practices

1. **Network Access**: Ensure your Ollama server is reachable from the gateway
2. **Model Availability**: Pull the required model before using the service
3. **Resource Management**: Monitor GPU/CPU usage on your Ollama server
4. **Security**: Use HTTPS and authentication if exposing Ollama over the internet
