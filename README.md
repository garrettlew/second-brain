
Requirements

- ollama
- chromadb


Local LLM setup (Mac M1)

install ollama. Download the ollama application (.dmg installer not the .sh script)

Set these variables before starting OLLAMA

launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
launchctl setenv OLLAMA_NUM_PARALLEL 1

If you have another machine (Mac M1) to run inference:
- On the inference machine, General > Sharing > Turn Remote login. Click the i icon and add the user to the Allow access for.
- On machine running script: ssh -N -L 11434:localhost:11434 <user>@<local_ip>
- ssh -N -L 11434:localhost:11434 garrett@192.168.1.1

Models:

- 'qwen3.5:9b' for the agents
- 'mxbai-embed-large' for embeddings


tail -f ~/.ollama/logs/server.log to see logs