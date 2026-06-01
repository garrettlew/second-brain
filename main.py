import argparse
import ollama

from Agent import Agent
from evaluation_helper import run_evaluation
from pathlib import Path
from Vault import Vault


def main(vault_path: str, inputfile: str, outputfile: str):
    print("Vault Path: {}".format(vault_path))
    model_client = ollama.Client(host="http://localhost:11434")
    agent = Agent(model_client)
    vault = Vault(vault_path, model_client, agent)
    if inputfile:
        print("Input File: {}".format(inputfile))
        current_note_filepath = Path(vault.vault_path) / inputfile
        current_note_content = current_note_filepath.read_text()

        input_setup = vault.get_note_setup_from_vault(inputfile)
        current_note_summary = input_setup.get("summary", "")
        current_note_tags = input_setup.get("tags", [])
        current_note_embedding = input_setup.get("embedding", [])

        candidate_notes = vault.query_related_notes_from_vault(
            note_id=inputfile,
            query_embedding=current_note_embedding,
            final_k=3
        )

        links = agent.run_linker_agents(current_note_tags, current_note_summary, current_note_content, candidate_notes)
        print(links)

        # if links:
        #     vault.append_links_to_note(inputfile, links)

    else:
        run_evaluation(vault, outputfile, agent.run_multi_agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that greets you.")
    parser.add_argument("--vaultpath", type=str, help="Absolute path to your vault.", required=True)
    parser.add_argument("--inputfile", type=str, help="The note to tag, summarize, and link related notes to.")
    parser.add_argument("--output", type=str, default="single_agent_results.csv")
    args = parser.parse_args()
    main(args.vaultpath, args.inputfile, args.output)
