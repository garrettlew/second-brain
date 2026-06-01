
# Second Brain

## Requirements

- ollama
- chromadb
- At least 12GB of RAM (to run locally)

## Local LLM setup 
(This was done on an M1 Mac)

1. Install ollama. Download the ollama application (.dmg installer not the .sh script).

2. (Optional for optimization) Set these variables before starting OLLAMA:
```shell
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
launchctl setenv OLLAMA_NUM_PARALLEL <MAX Parallel Requests>
```
Recommendation: OLLAMA_NUM_PARALLEL can be 1 if no parallel requests or 3 if there will be (top k candidates is 3 by default).

3. (Optional) If you have another machine (Mac M1) to run inference:
- On the inference machine, General > Sharing > Turn Remote login. Click the i icon and add the user to the Allow access for.

On machine running script: 
```shell
ssh -N -L 11434:localhost:11434 <user>@<local_ip>
EX: ssh -N -L 11434:localhost:11434 garrett@192.168.1.1
```

To get local IP address of machine:
```shell
ipconfig getifaddr en0
```

To get user:
```shell
whoami
```

To check if Ollama is running (done on the machine running Ollama or on the machine tunneling to the inference machine):
```shell
curl http://localhost:11434 
```

**Models**:

- 'qwen3.5:9b' for the agents. Uses about 8GB.
- 'mxbai-embed-large' for embeddings. Uses about 700MB.

To download the model without running it:
```shell
ollama pull <MODEL_NAME>
```

To show which models ollama has downloaded:
```shell
ollama list
```

To show which models ollama is running:
```shell
ollama ps
```

To see ollama logs:
```shell
tail -f ~/.ollama/logs/server.log
```

## Configurations

- cosine similarity is the recommended distance formula for mxbai-embed-large to use in chromaDB

## Running

To run the experiment on every note in the vault:
```shell
python3 main.py --vaultpath <PATH_TO_VAULT_FOLDER> --output <NAME_OF_OUTPUT_CSV.csv>

python3 main.py --vaultpath /Users/garrettlew/vault/ --output multi_agent_results.csv
```

To find links for a single given note:
```shell
python3 main.py --vaultpath <PATH_TO_VAULT_FOLDER> --inputfile <FILE_TO_GET_LINKS_FOR>

python3 main.py --vaultpath /Users/garrettlew/vault/ --inputfile example.md

```


## Data Sources

- **UCSC Vault**: Personal Obsidian notes from UCSC coursework (27 notes)
- **OMSCS Notes**: 15 notes selected from [m4ttsch/omscs-notes-notes](https://github.com/m4ttsch/omscs-notes-notes)
  (MIT License, © 2019 Matthew Schlenker). Used with permission per license terms.
