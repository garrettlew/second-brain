import argparse
import ollama

from Agent import Agent
from Vault import Vault
from evaluation_helper import run_evaluation


def single_agent_eval(vault_path, output_file):
    model_client = ollama.Client(host="http://localhost:11434")
    agent = Agent(model_client)
    vault = Vault(vault_path, model_client, agent)
    run_evaluation(vault, output_file, agent.run_single_agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vaultpath", type=str, required=True)
    parser.add_argument("--output", type=str, default="single_agent_results.csv")
    args = parser.parse_args()
    single_agent_eval(args.vaultpath, args.output)
