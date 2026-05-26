* # CSE 232B Project: Master Guide & Proposal Evaluation

  * ## Part 1: Proposal Evaluation

  * ### Overall Assessment: Strong Foundation with Fixable Gaps

  * Your proposal is well-structured and tackles a genuinely interesting research question grounded in recent literature (especially the Cemri et al. MAST taxonomy). The Obsidian note-enrichment testbed is a smart choice — it's concrete, demonstrable, and produces outputs that are easy to evaluate visually. Below are specific strengths and areas that need attention before you start coding.

  * ### Strengths

  * **Clear research framing.** You have two crisp research questions and a direct comparison (single-agent vs. multi-agent vs. multi-agent \+ overseer). The three-condition design lets you isolate the value of both decomposition and oversight.  
  *   
  * **Well-defined pipeline stages.** Each agent (Tagger, Summarizer, Linker) has explicit inputs, functions, and outputs. This makes implementation tractable and debugging straightforward.  
  *   
  * **Grounded related work.** You cite the right papers — MAST for failure taxonomy, RAGAS for RAG evaluation, LLM-as-judge for subjective scoring, and Reflexion/Self-Refine for the overseer motivation. This shows you understand the landscape.  
  *   
  * **Practical system.** The Obsidian \+ ChromaDB stack is lightweight, local-first, and avoids cloud dependencies that could bottleneck your timeline.

  * ### Issues to Address

  * **1\. Evaluation sample size is borderline (15-20 notes)**  
  *   
  * With only 15-20 notes and 3 experimental conditions, you'll have very limited statistical power. A single outlier note (e.g., one that's extremely short or off-topic) can skew your averages significantly. Consider expanding to 30-40 notes if possible, or at minimum, run each system 2-3 times per note to capture variance from LLM non-determinism (set temperature \> 0 and measure standard deviation).  
  *   
  * **2\. Tag Relevance as binary (0/1) loses information**  
  *   
  * A binary score can't distinguish between "completely hallucinated tags" and "relevant but slightly too broad." Consider a 3-point scale (0 \= irrelevant/hallucinated, 1 \= partially relevant or too broad, 2 \= highly relevant and specific) or keep binary but add a secondary metric for tag coverage (what fraction of "ground truth" tags were captured).  
  *   
  * **3\. Who is scoring? Inter-rater reliability is unaddressed**  
  *   
  * You mention qualitative metrics (Summary Faithfulness 1-3, Link Quality 1-3) but don't specify who scores them. If only one person scores, you have no measure of reliability. Either have at least 2 team members independently score and report Cohen's kappa, or use LLM-as-judge (which you cite) as a consistent automated scorer — but then you need to validate the LLM judge against a small human-scored sample.  
  *   
  * **4\. The three experimental conditions need explicit separation**  
  *   
  * Your evaluation plan says "both systems" but you actually have three conditions: (A) single agent, (B) multi-agent pipeline, (C) multi-agent \+ overseer. Make sure your evaluation table has three columns, not two. The overseer's impact is only visible when you compare B vs. C directly.  
  *   
  * **5\. Missing details that will bite you during implementation**  
  *   
  * **Which LLM?** GPT-4o, Claude, Llama 3, Mixtral? API vs. local? This affects cost, latency, and reproducibility.  
  * **Which embedding model?** For ChromaDB — OpenAI `text-embedding-3-small`, `all-MiniLM-L6-v2`, Nomic, etc.?  
  * **Cost tracking.** You measure latency but not token cost. For a course project, showing cost-per-note is a strong addition.  
  * **Note corpus.** Are these your own notes? Synthetic? From a public dataset? Reproducibility matters for the presentation.  
  *   
  * **6\. The Overseer design is ambitious — scope it carefully**  
  *   
  * The Overseer as described does three things: task routing, quality control (retry logic), and final synthesis. For a 4-week timeline, I'd recommend implementing quality control first (the most interesting and testable feature), task routing second (simple conditional logic), and final synthesis only if time permits.  
  *   
  * ---

  * ## Part 2: Step-by-Step Master Guide

  * ### Timeline Overview (May 8 \- June 4, 2026\)

| Phase | Dates | Goal |
| :---- | :---- | :---- |
| Phase 1: Setup & Data | May 8 \- 14 | Repo, environment, note corpus, ChromaDB populated |
| Phase 2: Core Agents | May 15 \- 21 | Single agent \+ 3-stage pipeline working end-to-end |
| Phase 3: Overseer & Eval | May 22 \- 28 | Overseer agent, evaluation framework, run experiments |
| Phase 4: Analysis & Demo | May 29 \- June 4 | Results analysis, demo, presentation, final report |

  *   
  * ---

  * ### Phase 1: Setup & Data Preparation (May 8-14)

  * #### Day 1-2: Project Infrastructure

  * **1\. Initialize the repository**  
  * 

```
project-root/
├── README.md
├── requirements.txt
├── .env                    # API keys (gitignored)
├── config.yaml             # Model names, temperature, top-k, etc.
├── vault/                  # Obsidian test notes (markdown files)
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py       # Abstract base class for all agents
│   │   ├── single_agent.py     # Condition A: monolithic agent
│   │   ├── tagger.py           # Stage 1
│   │   ├── summarizer.py       # Stage 2
│   │   ├── linker.py           # Stage 3
│   │   └── overseer.py         # Stage 4
│   ├── orchestrator.py         # Pipeline control plane
│   ├── vector_store.py         # ChromaDB wrapper
│   ├── note_parser.py          # Read/write Obsidian markdown
│   └── utils.py                # Logging, cost tracking, timing
├── eval/
│   ├── run_evaluation.py       # Main eval script
│   ├── metrics.py              # Scoring functions
│   ├── judge.py                # LLM-as-judge implementation
│   └── results/                # Output CSVs and charts
├── demo/
│   └── app.py                  # Streamlit or Gradio demo
└── presentation/
    └── slides.pptx
```

  *   
  * **2\. Set up the Python environment**  
  * 

```shell
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install openai anthropic chromadb sentence-transformers
pip install pyyaml python-dotenv pandas matplotlib seaborn
pip install streamlit  # for demo
```

  *   
  * **3\. Create config.yaml**  
  * 

```
llm:
  provider: "openai"        # or "anthropic", "local"
  model: "gpt-4o-mini"      # cheaper for development, switch to gpt-4o for final runs
  temperature: 0.3           # low for reproducibility, >0 for variance measurement
  max_tokens: 1024

embedding:
  model: "all-MiniLM-L6-v2" # local, free, fast
  # model: "text-embedding-3-small"  # OpenAI alternative

chromadb:
  collection_name: "obsidian_notes"
  persist_directory: "./chroma_db"
  top_k: 5                  # retrieve top 5, linker picks top 3

evaluation:
  num_runs: 3               # runs per note per condition for variance
```

  * #### Day 3-4: Prepare the Note Corpus

  * This is critical and often underestimated. Your evaluation is only as good as your test data.  
  *   
  * **Curate 30-40 notes with deliberate variety:**  
  *   
  * 8-10 short notes (\< 100 words) — tests edge cases for summarizer  
  * 10-12 medium notes (100-500 words) — the bread-and-butter case  
  * 8-10 long notes (500+ words) — tests whether agents handle complexity  
  * 4-6 notes on overlapping topics — tests whether the linker finds connections  
  * 2-3 notes on completely unrelated topics — tests whether the linker avoids false links  
  *   
  * **For each note, create ground truth annotations (a separate JSON file):**  
  * 

```json
{
  "note_id": "quantum_computing_basics.md",
  "ground_truth_tags": ["quantum-computing", "qubits", "superposition", "CS-theory"],
  "key_concepts": ["qubit definition", "superposition principle", "entanglement"],
  "expected_links": ["linear_algebra_notes.md", "physics_fundamentals.md"],
  "difficulty": "medium",
  "word_count": 342
}
```

  * #### Day 5-6: Build the Vector Store

```py
# src/vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer
import os, glob

class NoteVectorStore:
    def __init__(self, config):
        self.client = chromadb.PersistentClient(path=config["persist_directory"])
        self.collection = self.client.get_or_create_collection(
            name=config["collection_name"],
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer(config["embedding"]["model"])

    def index_vault(self, vault_path: str):
        """Index all markdown files in the Obsidian vault."""
        notes = glob.glob(os.path.join(vault_path, "*.md"))
        for note_path in notes:
            with open(note_path, 'r') as f:
                content = f.read()
            note_id = os.path.basename(note_path)
            embedding = self.embedder.encode(content).tolist()
            self.collection.upsert(
                ids=[note_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{"filename": note_id, "word_count": len(content.split())}]
            )

    def query(self, text: str, top_k: int = 5, exclude_id: str = None):
        """Retrieve top-k similar notes, excluding the query note itself."""
        embedding = self.embedder.encode(text).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k + 1  # fetch extra in case we need to filter
        )
        # Filter out the note itself
        filtered = []
        for i, doc_id in enumerate(results['ids'][0]):
            if doc_id != exclude_id:
                filtered.append({
                    "id": doc_id,
                    "document": results['documents'][0][i],
                    "distance": results['distances'][0][i]
                })
        return filtered[:top_k]
```

  *   
  * **Milestone check (end of Phase 1):** You should be able to run `python -c "from src.vector_store import NoteVectorStore; ..."` and get query results from your indexed vault.  
  *   
  * ---

  * ### Phase 2: Core Agent Implementation (May 15-21)

  * #### Day 7-8: Base Agent & Single Agent (Condition A)

```py
# src/agents/base_agent.py
import time
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, llm_client, model: str, system_prompt: str):
        self.llm = llm_client
        self.model = model
        self.system_prompt = system_prompt
        self.last_usage = None  # token tracking

    def call(self, user_message: str) -> dict:
        start = time.time()
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}  # enforce structured output
        )
        elapsed = time.time() - start
        self.last_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "latency_seconds": elapsed
        }
        return self._parse_response(response.choices[0].message.content)

    @abstractmethod
    def _parse_response(self, raw: str) -> dict:
        pass
```

  * 

```py
# src/agents/single_agent.py
import json
from .base_agent import BaseAgent

SINGLE_AGENT_PROMPT = """You are an Obsidian note enrichment assistant. Given a markdown note,
perform ALL of the following tasks and return a JSON object:

1. "tags": an array of 3-7 relevant, specific topical tags for this note
2. "summary": a 2-3 sentence summary capturing the core concepts
3. "suggested_links": an array of up to 3 objects, each with:
   - "note_title": the filename of a related note
   - "justification": one sentence explaining the connection

Context: The user's vault contains these notes: {vault_index}

Return ONLY valid JSON. No markdown fencing."""

class SingleAgent(BaseAgent):
    def __init__(self, llm_client, model, vault_index):
        prompt = SINGLE_AGENT_PROMPT.replace("{vault_index}", vault_index)
        super().__init__(llm_client, model, prompt)

    def _parse_response(self, raw):
        return json.loads(raw)
```

  * #### Day 9-10: Multi-Agent Pipeline (Condition B)

  * Implement each specialized agent with a narrow, focused system prompt. The key principle: each agent does ONE thing well.  
  *   
  * **Tagger Agent:**  
  * 

```py
TAGGER_PROMPT = """You are a note classification specialist. Given a markdown note,
extract 3-7 highly specific, relevant tags. Tags should be:
- Specific (prefer "gradient-descent" over "machine-learning")
- Consistent with an academic knowledge base
- A mix of topical tags and categorical tags (e.g., "concept", "tutorial", "reference")

Return JSON: {"tags": ["tag1", "tag2", ...]}"""
```

  *   
  * **Summarizer Agent:**  
  * 

```py
SUMMARIZER_PROMPT = """You are a note summarization specialist. Given a markdown note
and a set of tags that describe it, write a 2-3 sentence summary that:
- Captures the core concepts faithfully (no hallucinated information)
- Uses the tags as grounding context to emphasize the most relevant themes
- Is concise enough to scan in a sidebar preview

Return JSON: {"summary": "..."}"""
```

  *   
  * **Linker Agent:**  
  * 

```py
LINKER_PROMPT = """You are a knowledge-graph linking specialist. Given a note's summary,
its tags, and a list of candidate related notes (retrieved by semantic similarity),
select the top 3 most meaningfully connected notes.

For each link, provide a one-line justification explaining the thematic connection.
Prefer deep, non-obvious connections over surface-level keyword overlap.

Return JSON: {"links": [{"note_title": "...", "justification": "..."}, ...]}"""
```

  *   
  * **Orchestrator:**  
  * 

```py
# src/orchestrator.py
class PipelineOrchestrator:
    def __init__(self, tagger, summarizer, linker, vector_store):
        self.tagger = tagger
        self.summarizer = summarizer
        self.linker = linker
        self.vector_store = vector_store
        self.run_log = []  # track tokens + latency per stage

    def process_note(self, note_content: str, note_id: str) -> dict:
        # Stage 1: Tag
        tag_result = self.tagger.call(note_content)
        self.run_log.append(("tagger", self.tagger.last_usage))

        # Stage 2: Summarize (receives note + tags)
        summary_input = f"Note:\n{note_content}\n\nTags: {tag_result['tags']}"
        summary_result = self.summarizer.call(summary_input)
        self.run_log.append(("summarizer", self.summarizer.last_usage))

        # Stage 3: Link (receives summary + tags + retrieved candidates)
        candidates = self.vector_store.query(
            summary_result['summary'], top_k=5, exclude_id=note_id
        )
        link_input = (
            f"Summary: {summary_result['summary']}\n"
            f"Tags: {tag_result['tags']}\n"
            f"Candidate notes:\n" +
            "\n".join(f"- {c['id']}: {c['document'][:200]}..." for c in candidates)
        )
        link_result = self.linker.call(link_input)
        self.run_log.append(("linker", self.linker.last_usage))

        return {
            "tags": tag_result["tags"],
            "summary": summary_result["summary"],
            "links": link_result["links"],
            "usage": list(self.run_log)
        }
```

  * #### Day 11-12: Integration Testing

  * Before moving on, verify that both Condition A and Condition B produce valid outputs on 3-5 test notes. Check for:  
  *   
  * JSON parsing errors (add try/except with retries)  
  * Tags that are actually strings (not nested objects)  
  * Summaries that are 2-3 sentences (not 2-3 paragraphs)  
  * Links that reference real notes in the vault (not hallucinated filenames)  
  * Latency is reasonable (\< 30s per note for single agent, \< 60s for pipeline)  
  *   
  * Write a simple smoke test:  
  * 

```py
# tests/test_smoke.py
def test_single_agent_output_schema():
    result = single_agent.call(sample_note)
    assert "tags" in result and isinstance(result["tags"], list)
    assert "summary" in result and len(result["summary"]) > 20
    assert "suggested_links" in result

def test_pipeline_output_schema():
    result = orchestrator.process_note(sample_note, "test.md")
    assert len(result["tags"]) >= 3
    assert len(result["summary"].split(". ")) >= 2
    assert len(result["links"]) <= 3
```

  *   
  * **Milestone check (end of Phase 2):** Both the single agent and the 3-stage pipeline should process any note from your vault and return valid, structured output.  
  *   
  * ---

  * ### Phase 3: Overseer & Evaluation (May 22-28)

  * #### Day 13-14: Overseer Agent (Condition C)

  * Start with quality control (the highest-impact feature), then add task routing.  
  * 

```py
# src/agents/overseer.py
OVERSEER_PROMPT = """You are a quality-control overseer for a note enrichment pipeline.
You will receive the original note along with the outputs from three specialist agents
(tags, summary, and suggested links).

Your job:
1. CHECK TAG QUALITY: Are all tags relevant and specific? Flag any that are hallucinated,
   overly broad, or inconsistent with the note content. Suggest replacements if needed.
2. CHECK SUMMARY FAITHFULNESS: Does the summary accurately reflect the note?
   Does it introduce any facts not present in the original? Flag issues.
3. CHECK LINK COHERENCE: Do the suggested links and their justifications make sense
   given the tags and summary? Flag weak connections.
4. OVERALL COHERENCE: Are the tags, summary, and links consistent with each other?

Return JSON:
{
  "approved": true/false,
  "revised_tags": [...] or null (if tags are fine),
  "revised_summary": "..." or null,
  "revised_links": [...] or null,
  "issues_found": ["description of each issue"],
  "retry_recommended": true/false
}"""
```

  *   
  * **Task routing (simple conditional logic in the orchestrator):**  
  * 

```py
def should_skip_summary(self, note_content: str) -> bool:
    """Skip summarization for very short notes (< 50 words)."""
    return len(note_content.split()) < 50

def process_note_with_overseer(self, note_content, note_id, max_retries=1):
    result = self.process_note(note_content, note_id)  # run base pipeline

    # Overseer review
    overseer_input = f"""Original note:\n{note_content}\n
    Tags: {result['tags']}\nSummary: {result['summary']}\nLinks: {result['links']}"""

    review = self.overseer.call(overseer_input)

    if not review["approved"] and review["retry_recommended"] and max_retries > 0:
        # Re-run with corrective context
        return self.process_note_with_overseer(note_content, note_id, max_retries - 1)

    # Apply any revisions from the overseer
    if review.get("revised_tags"):
        result["tags"] = review["revised_tags"]
    if review.get("revised_summary"):
        result["summary"] = review["revised_summary"]
    if review.get("revised_links"):
        result["links"] = review["revised_links"]

    result["overseer_review"] = review
    return result
```

  * #### Day 15-17: Evaluation Framework

  * **Build the evaluation harness:**  
  * 

```py
# eval/run_evaluation.py
import json, csv, time
from pathlib import Path

def run_full_evaluation(conditions, notes, num_runs=3):
    """
    conditions: dict mapping name -> callable(note_content, note_id) -> result
    notes: list of (note_id, note_content, ground_truth) tuples
    """
    all_results = []

    for condition_name, process_fn in conditions.items():
        for note_id, content, truth in notes:
            for run_idx in range(num_runs):
                start = time.time()
                result = process_fn(content, note_id)
                total_latency = time.time() - start

                all_results.append({
                    "condition": condition_name,
                    "note_id": note_id,
                    "run": run_idx,
                    "tags": result["tags"],
                    "summary": result["summary"],
                    "links": result.get("links", []),
                    "latency_seconds": total_latency,
                    "ground_truth": truth
                })

    return all_results
```

  *   
  * **LLM-as-Judge scoring (automate what you can):**  
  * 

```py
# eval/judge.py
JUDGE_PROMPT = """You are an impartial evaluator for a note enrichment system.
Given the original note and the system's output, score the following:

1. Tag Relevance (0-2 per tag, then average):
   0 = irrelevant or hallucinated
   1 = partially relevant or too broad
   2 = highly relevant and specific

2. Summary Faithfulness (1-3):
   1 = misses key point or hallucinates facts
   2 = adequate, captures basic premise but misses nuance
   3 = highly faithful, captures core concepts perfectly

3. Link Quality (1-3 per link, then average):
   1 = irrelevant connection
   2 = plausible but superficial
   3 = highly useful, reveals meaningful thematic link

Return JSON:
{
  "tag_scores": [score_per_tag],
  "tag_avg": float,
  "summary_faithfulness": int,
  "link_scores": [score_per_link],
  "link_avg": float,
  "reasoning": "brief explanation of scores"
}"""
```

  *   
  * **Important:** Validate the LLM judge by having 2 human team members independently score a random subset of 10 note-outputs. Compare human scores to LLM-judge scores and report correlation. This takes about 1-2 hours and massively strengthens your evaluation section.

  * #### Day 17-18: Run All Experiments

```py
conditions = {
    "single_agent": lambda content, nid: single_agent.call(content),
    "multi_agent": lambda content, nid: orchestrator.process_note(content, nid),
    "multi_agent_overseer": lambda content, nid: orchestrator.process_note_with_overseer(content, nid)
}

results = run_full_evaluation(conditions, note_corpus, num_runs=3)
```

  *   
  * Log everything: raw outputs, token counts, latency, and cost per call. You'll need this for the presentation.  
  *   
  * **Milestone check (end of Phase 3):** You should have a complete CSV of results across all three conditions, all notes, and all runs, plus LLM-judge scores for every output.  
  *   
  * ---

  * ### Phase 4: Analysis, Demo & Presentation (May 29 \- June 4\)

  * #### Day 19-20: Results Analysis

  * **Generate comparison tables and charts:**  
  * 

```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("eval/results/all_results.csv")

# Mean scores by condition
summary_table = df.groupby("condition").agg({
    "tag_avg": ["mean", "std"],
    "summary_faithfulness": ["mean", "std"],
    "link_avg": ["mean", "std"],
    "latency_seconds": ["mean", "std"]
}).round(3)

# Box plots for each metric
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, metric in zip(axes, ["tag_avg", "summary_faithfulness", "link_avg", "latency_seconds"]):
    sns.boxplot(data=df, x="condition", y=metric, ax=ax)
    ax.set_title(metric.replace("_", " ").title())
plt.tight_layout()
plt.savefig("eval/results/comparison_boxplots.png", dpi=150)
```

  *   
  * **Key analyses to include:**  
  *   
1. Mean \+ std for each metric per condition (the core comparison table)  
2. Box plots or violin plots showing score distributions  
3. Per-note breakdown: are there note types where MAS wins vs. loses?  
4. Latency vs. quality scatter plot (is the extra time worth it?)  
5. Cost analysis: total tokens and estimated API cost per condition  
6. Overseer impact: how often did it change outputs, and did changes improve scores?  
7. Failure mode analysis: categorize bad outputs using MAST taxonomy categories from Cemri et al.

   * #### Day 21-22: Build the Demo

   * Use Streamlit for a fast, polished interactive demo:  
   * 

```py
# demo/app.py
import streamlit as st

st.title("Single Agent vs. Multi-Agent Note Enrichment")

# Sidebar: select a note and condition
note = st.sidebar.selectbox("Select a note", note_files)
condition = st.sidebar.radio("Architecture", ["Single Agent", "Multi-Agent", "Multi-Agent + Overseer"])

# Main area: show the note and enrichment side-by-side
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Note")
    st.markdown(note_content)

with col2:
    st.subheader("Enriched Output")
    if st.button("Run Enrichment"):
        with st.spinner("Processing..."):
            result = process(note_content, condition)
        st.write("**Tags:**", result["tags"])
        st.write("**Summary:**", result["summary"])
        st.write("**Suggested Links:**")
        for link in result["links"]:
            st.write(f"- {link['note_title']}: {link['justification']}")

# Show metrics comparison
st.subheader("Evaluation Results")
st.dataframe(summary_table)
st.image("eval/results/comparison_boxplots.png")
```

   * #### Day 23-25: Presentation & Final Polish

   * **Suggested slide structure (12-15 slides, \~15 min talk):**  
   *   
1. Title slide  
2. Motivation: why multi-agent systems? What does MAST tell us about their failures?  
3. Research questions (2 questions, clearly stated)  
4. System architecture diagram (single agent vs. pipeline vs. pipeline \+ overseer)  
5. Obsidian testbed and note corpus overview  
6. Technical details: ChromaDB, embedding model, LLM choice, prompt design  
7. Evaluation methodology: metrics, LLM-as-judge, human validation  
8. Results table: mean scores across conditions  
9. Results visualization: box plots or bar charts  
10. Failure mode analysis: what went wrong, mapped to MAST categories  
11. Overseer impact: did quality control help? When did it hurt?  
12. Key findings and answers to research questions  
13. Limitations and future work  
14. Live demo (switch to Streamlit app)  
15. References  
    *   
    * **Final checklist before submission:**  
    *   
- [ ] All code runs end-to-end from a clean environment (test with `pip install -r requirements.txt`)  
- [ ] README.md has clear setup and run instructions  
- [ ] Results are reproducible (fixed random seeds, config file committed)  
- [ ] Demo runs locally without API key issues (pre-cache some results as fallback)  
- [ ] Presentation rehearsed at least once with timing  
- [ ] Raw data (CSVs, JSONs) committed to repo for verification  
      *   
      * ---

      * ## Part 3: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
| :---- | :---- | :---- | :---- |
| API rate limits or outages | Medium | High | Cache all API responses; have a local LLM fallback (e.g., Ollama \+ Llama 3 8B) |
| ChromaDB indexing issues | Low | Medium | Test early with 5 notes; keep a backup flat-file similarity search |
| Overseer creates infinite retry loops | Medium | Medium | Hard cap retries at 1; log all overseer decisions |
| LLM-as-judge scores are inconsistent | Medium | High | Validate against human scores on a 10-note subset; report inter-rater agreement |
| Notes are too uniform for meaningful comparison | Medium | High | Deliberately vary note length, topic, and complexity; include edge cases |
| Demo breaks during presentation | Low | High | Pre-record a backup video of the demo; cache example outputs |
| Team coordination bottlenecks | Medium | Medium | Assign clear ownership per module; merge to main daily |

      *   
      * ---

      * ## Part 4: Suggested Team Task Division (3-4 members)

| Member | Primary Responsibility | Secondary |
| :---- | :---- | :---- |
| Member A | Note corpus curation \+ ground truth annotations | Evaluation scoring \+ analysis |
| Member B | Vector store (ChromaDB) \+ Linker agent | Demo (Streamlit) |
| Member C | Single agent \+ Tagger \+ Summarizer | Integration testing |
| Member D | Orchestrator \+ Overseer agent | Presentation slides |

      *   
      * If you're a 3-person team, merge Member A and D responsibilities.  
      *   
      * ---

      * ## Part 5: Quick Reference — Key Decisions to Make This Week

1. **Which LLM API?** Recommendation: start with `gpt-4o-mini` for development (cheap, fast), switch to `gpt-4o` for final evaluation runs. Budget approximately $5-15 total.  
2. **Which embedding model?** Recommendation: `all-MiniLM-L6-v2` via sentence-transformers (free, local, no API dependency).  
3. **Note corpus source?** Best option: use real notes from a team member's vault. Second best: curate notes from Wikipedia or course materials. Avoid purely synthetic notes.  
4. **Evaluation approach?** Recommendation: LLM-as-judge (GPT-4o) for all outputs \+ human scoring on a 10-note validation subset.  
5. **Temperature setting?** Use 0.3 for consistency. Run each condition 3 times to measure variance.

