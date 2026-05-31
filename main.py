import argparse
import concurrent.futures
import json
import ollama

from evaluation_helper import run_evaluation
from pathlib import Path
from pydantic import BaseModel
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

        links = run_linker_agents(agent, current_note_tags, current_note_summary, current_note_content, candidate_note_results)
        print(links)

        # if links:
        #     vault.append_links_to_note(inputfile, links)

    else:
        run_evaluation(agent, model_client, vault, args.output, run_multi_agent)


def run_multi_agent(model_client, raw_input_note, candidate_notes):
    agent = Agent(model_client)
    tags = agent.tagger_agent(raw_input_note)
    summary = agent.summarizer_agent(raw_input_note, tags)
    links = run_linker_agents(agent, tags, summary, raw_input_note, candidate_notes)
    result = {
        "tags": tags,
        "summary": summary,
        "links": links
    }
    return result

def run_linker_agents(agent, current_note_tags, current_note_summary, current_note_content, candidate_note_results):
    """
    Runs linker agent calls in parallel for each candidate note and returns relevant links.

    Skips the first candidate result (index 0) as it is the input note itself.

    Args:
        agent: The Agent instance used to call linker_agent.
        current_note_tags (list[str]): Tags for the current note.
        current_note_summary (str): Summary of the current note.
        current_note_content (str): Raw text content of the current note.
        candidate_note_results (dict): Query results from the vector DB containing
            ids, documents, and metadatas for candidate notes.

    Returns:
        list[dict]: A list of relevant links, each with keys:
            - "id": the candidate note's ID
            - "reason": one sentence explaining the connection
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                agent.linker_agent,
                current_note_tags,
                current_note_summary,
                current_note_content,
                candidate_note_results[i]["tags"],
                candidate_note_results[i]["summary"]
            ): candidate_note_results[i]["note_title"]
            for i in range(len(candidate_note_results))
        }

    links = []
    for future in concurrent.futures.as_completed(futures):
        candidate_id = futures[future]
        judgement = future.result()
        print(f"Future {candidate_id} returned. Judgement: {judgement}.")
        if judgement.relevant:
            links.append({"id": candidate_id, "reason": judgement.reason})

    return links

class Judgement(BaseModel):
    relevant: bool
    reason: str


class Agent:
    def __init__(self, model_client, model_type='qwen3.5:9b'):
        self.model_client = model_client
        self.model_type = model_type

    def model_chat(self, messages: list[dict[str, str]], output_format=None, think=False):
        chat_response = self.model_client.chat(
            model=self.model_type,
            messages=messages,
            format=output_format,
            stream=False,
            think=think
        )
        return chat_response


    def tagger_agent(self, note_text: str) -> list[str]:
        TAGGER_SYSTEM_PROMPT = """
        You are a note tagging agent. Your only job is to read a note and return relevant tags.

        Rules:
        - Return 3 tags
        - Tags should be lowercase, hyphenated (e.g. machine-learning, not Machine Learning)
        - Be specific but not overly narrow
        - Return ONLY a JSON list of strings [str, str, str], nothing else

        Example output:
        ["machine-learning", "meeting", "attention-mechanism"]
        """

        response = self.model_chat(
            messages=[
                {"role": "system", "content": TAGGER_SYSTEM_PROMPT},
                {"role": "user", "content": note_text}
            ],
            output_format='json'
        )
        print(response)
        raw = response["message"]["content"]
        tags = json.loads(raw)
        return tags

    def summarizer_agent(self, note_text: str, tags: list[str]) -> str:
        SUMMARIZER_SYSTEM_PROMPT = """
        You are a note summarization agent. Your job is to write a concise summary of a note.

        Rules:
        - Write at most 2-3 sentences
        - Focus on the core idea or insight of the note, not peripheral details
        - Use the provided tags as a guide for what the note is primarily about
        - Do not include opinions or evaluation of the content
        - Return ONLY the summary text, nothing else — no preamble, no labels

        Example output:
        The attention mechanism is the key idea in transformer models and it allows the model to weigh the relevance of different tokens. There are two types of attention: self-attention and cross-attention, and they're calculated using a scaled dot-product operation. The original 'Attention Is All You Need' paper is a key source.
        """

        user_message = f"""Note:
        {note_text}

        Tags identified for this note: {', '.join(tags)}
        """

        response = self.model_chat(
            messages=[
                {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        return response["message"]["content"]

    def linker_agent(self, current_note_tags: list[str], current_note_summary: str, current_note_content: str, candidate_note_tags: list[str], candidate_note_summary: str) -> list[dict]:
        LINKER_SYSTEM_PROMPT = """
        You are a note linking agent. Your job is to decide if the provided candidate note is
        genuinely relevant to link to the current note.

        Rules:
        - Only set relevant to true if a candidate has a meaningful conceptual connection to the current note
        - Reject candidates that are only superficially or tangentially related
        - For a kept candidate, write one sentence explaining the connection
        - You must respond with ONLY this exact JSON structure:
            {
                "relevant": true or false,
                "reason": "your one sentence reason here"
            }

        Relevant example output:
            {"relevant": true, "reason": "Both notes discuss attention mechanisms in neural networks"}

        Irrelevant example output:
            {"relevant": false, "reason": "Not related as the candidate note is about fence post embeddings while the current note is about the embeddings output of transformer encoders"}
        """

        user_message = f"""Current note:
            Tags: {current_note_tags}
            Summary: {current_note_summary}
            Full text: {current_note_content}

            Candidate:
            Tags: {candidate_note_tags}
            Summary: {candidate_note_summary}

            Decide if the candidate is genuinely worth linking to the current note ("relevant": True) and give a reason why."""

        response = self.model_chat(
            messages=[
                {"role": "system", "content": LINKER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            output_format=Judgement.model_json_schema()
        )
        parsed = Judgement.model_validate_json(response.message.content)
        return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that greets you.")
    parser.add_argument("--vaultpath", type=str, help="Absolute path to your vault.", required=True)
    parser.add_argument("--inputfile", type=str, help="The note to tag, summarize, and link related notes to.")
    parser.add_argument("--output", type=str, default="single_agent_results.csv")
    args = parser.parse_args()
    main(args.vaultpath, args.inputfile)