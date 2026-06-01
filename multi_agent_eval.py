import argparse
import ollama

from Agent import Agent
from evaluation_helper import run_evaluation
from Vault import Vault


def multi_agent_eval(vault_path: str, outputfile: str):
    print("Vault Path: {}".format(vault_path))
    model_client = ollama.Client(host="http://localhost:11434")
    agent = Agent(model_client)
    vault = Vault(vault_path, model_client, agent)

    run_evaluation(vault, outputfile, agent.run_multi_agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that greets you.")
    parser.add_argument("--vaultpath", type=str, help="Absolute path to your vault.", required=True)
    parser.add_argument("--output", type=str, default="multi_agent_results.csv")
    args = parser.parse_args()
    multi_agent_eval(args.vaultpath, args.output)
