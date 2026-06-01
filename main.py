import argparse
import ollama

from Agent import Agent
from evaluation_helper import run_evaluation
from pathlib import Path
from Vault import Vault


def main(vault_path: str, inputfile: str):
    print("Vault Path: {}".format(vault_path))
    model_client = ollama.Client(host="http://localhost:11434")
    agent = Agent(model_client)
    vault = Vault(vault_path, model_client, agent)
    if inputfile:
        print("Input File: {}".format(inputfile))
        current_note_filepath = Path(vault.vault_path) / inputfile
        current_note_content = current_note_filepath.read_text()

        current_note_results = vault.collection.get(
            ids=[inputfile],
            include=["embeddings", "documents", "metadatas"]
        )

        current_note_summary = current_note_results['documents'][0]
        current_note_metadata = current_note_results['metadatas'][0]
        current_note_tags = current_note_metadata.get('tags')

        candidate_note_results = vault.collection.query(
            query_embeddings=[current_note_results['embeddings'][0]],
            n_results=4  # ask for 4, discard the first (self)
        )

        links = agent.run_linker_agents(current_note_tags, current_note_summary, current_note_content, candidate_note_results)
        print(links)

        # if links:
        #     vault.append_links_to_note(inputfile, links)

    else:
        run_evaluation(agent, model_client, vault, args.output, run_multi_agent)


def run_multi_agent(model_client, raw_input_note, candidate_notes):
    agent = Agent(model_client)
    tags = agent.tagger_agent(raw_input_note)
    summary = agent.summarizer_agent(raw_input_note, tags)
    links = agent.run_linker_agents(agent, tags, summary, raw_input_note, candidate_notes)
    result = {
        "tags": tags,
        "summary": summary,
        "links": links
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that greets you.")
    parser.add_argument("--vaultpath", type=str, help="Absolute path to your vault.", required=True)
    parser.add_argument("--inputfile", type=str, help="The note to tag, summarize, and link related notes to.")
    parser.add_argument("--output", type=str, default="single_agent_results.csv")
    args = parser.parse_args()
    main(args.vaultpath, args.inputfile)
