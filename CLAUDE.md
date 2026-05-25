# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Second Brain is a local AI-powered note indexing tool for Obsidian-style Markdown vaults. It uses locally-hosted LLMs (via Ollama) to auto-tag and summarize notes, then stores the resulting embeddings in a ChromaDB vector database for semantic retrieval.

## Running

```shell
source .venv/bin/activate
python3 main.py --vaultpath <PATH_TO_VAULT_FOLDER> --inputfile <FILE_TO_GET_LINKS_FOR>
```

`--inputfile` is optional. `--vaultpath` must be an absolute path to a folder of `.md` files.

## Dependencies

Dependencies are managed via the `.venv` virtual environment (no `requirements.txt`). Key packages:
- `ollama` (0.6.2) — Python client for the local Ollama inference server
- `chromadb` (1.5.9) — persistent local vector database

## Architecture

Everything lives in `main.py`. Three components:

**`Agent`** — wraps the Ollama chat API and provides two specialized agents:
- `tagger_agent(note_text)` → `list[str]`: calls the LLM with a structured prompt to return exactly 3 JSON tags
- `summarizer_agent(note_text, tags)` → `str`: uses the tags as context to generate a 2–3 sentence summary

**`Vault`** — manages the ChromaDB collection (`"second-brain"`) and vault indexing:
- On init, calls `index_vault()` which walks all `.md` files under `vault_path`, skipping files already present in the collection (by filename as ID)
- Each indexed note goes through `Agent` for tags + summary, then the summary is embedded with `mxbai-embed-large` and stored with `{tags, last_modified}` metadata
- ChromaDB is stored persistently in the default local path (current directory)

**`main()`** — wires together an `Agent` and a `Vault`. Currently contains hardcoded test text (a Wikipedia excerpt) rather than reading from `--inputfile`; the vault indexing is the live code path.

## Local LLM Setup (Ollama)

Two models must be pulled before running:
```shell
ollama pull qwen3.5:9b        # agent model, ~8GB VRAM
ollama pull mxbai-embed-large # embedding model, ~700MB
```

Verify Ollama is running: `curl http://localhost:11434`

To tunnel inference from a second machine:
```shell
ssh -N -L 11434:localhost:11434 <user>@<local_ip>
```

Optional performance flags (set before launching Ollama on macOS):
```shell
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
launchctl setenv OLLAMA_NUM_PARALLEL 1
```

## Known Limitations / Active TODOs

- `main()` uses hardcoded test text instead of reading `--inputfile`
- `index_vault()` only skips re-indexing by filename — it doesn't detect modified files (compare against `last_modified` metadata to implement incremental updates)
- The `Vault` constructor has commented-out code to drop and recreate the collection; uncomment to do a full reindex
- No link suggestion logic is implemented yet (the `--inputfile` argument exists for this future feature)
