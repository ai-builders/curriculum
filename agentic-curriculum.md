> **Goals:** for each student to understand and develop a fully fledged agent with working internal and external memory, RAG retrieval, and programmatic tool use to perform Google searches and web retrieval. The notebooks will be designed to promote scientific thinking via benchmarking and metric design rather than a simple LLM call. Each week imports the prior week's module rather than starting fresh for a sense of continuity.

---

# Week 1 — The Agentic Loop

## Goal

- Understand what formally distinguishes an _agent_ from a single-shot LLM call
- Implement a ReAct loop from scratch without any agentic framework
- Build `ChatAgent`,the base class that Weeks 2 and 3 will extend
- Identify and reproduce common failure modes (infinite loops, context collapse, prompt injection)

## Readings

- [ReAct: Synergizing Reasoning and Acting in Language Models — Yao et al. 2022](https://arxiv.org/abs/2210.03629) The core paper
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models — Wei et al. 2022](https://arxiv.org/abs/2201.11903) Prerequisite - CoT
- Probability Theory (Markov chains)

## Lecture

### Block 1 — From LLM to Agent (20 min)

- Single-shot inference vs. sequential decision making: $f(x) \to y$ vs. $\pi(a_t \mid s_t)$
- The agent loop: perception → reasoning → action → memory → repeat
- Formal definition: trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$
- The LLM as a _frozen policy_: instead of training like RL, we use prompting to optimize the LLM
- Instead of wasting *training time*, LLM performs on the currency of *tokens* in exchange for some *success metric*

### Block 2 — ReAct: a scientific framework for agents (30 min)

- **Motivation**: Introduce students to CoT prompting and why it fails to execute multi-step tasks (no grounding and reasoning)
- **ReAct**: interleaves **Thought** → **Action** → **Observation** → repeat
- Walk through a paper example: multi-hop Wikipedia QA on the board
- The stopping condition as a formal predicate: $\delta(s_t) \in {0, 1}$ — why it is non-negotiable (infinite loop)
- Introduce more general form of ReAct (i.e. one based off Tree-of-Thought, wider search, improves exploration at the cost of tokens)

### Block 3 — Memory as state (25 min)

- **Motivation**: in-context memory, the conversation window **is** $s_t$, append next chat to the previous -> $|s_t| > L_{\max}$ — what gets dropped?
- **Two compression strategies**: naive truncation (drop oldest) vs. rolling summary (LLM summarises oldest $k$ turns)
- Four memory types taxonomy: plant the seed for Week 2
    - **In-context** (working memory, this week)
    - **External / retrieval** (Week 2)
    - **Parametric** (baked into weights, not addressable as it requires SFT)
    - **Episodic** (stored trajectories, Week 3)
- "Lost in the middle" problem: attention degrades for content in the middle of long contexts — empirical result to show students

### Block 4 — Safety & Pitfalls: Week 1 (15 min)

- **Infinite loops:** agent never emits $\delta = 1$; always enforce `max_iterations`
- **Prompt injection (introduce early):** user input contains `"Ignore previous instructions..."` demo live with "dumb" models
- **Sycophancy / context collapse:** LLM agrees with its own prior outputs over time; show a concrete example
- **Lost in the middle:** plant a fact at position 1 vs. position $T/2$ and compare retrieval accuracy

## Example Notebook — `agent_week1.ipynb`

- Part 1 — Implement the ReAct loop**
    - Build `class ChatAgent(messages, system_prompt, max_iter)` with raw API calls (no framework)
    - Toy environment: a Python `dict` acting as a knowledge base the agent can `lookup(key)`
    - Visualize the trajectory with `networkx`: nodes = states, edge labels = actions taken
- Part 2 — Memory compression benchmark**
    - Synthetic conversation: plant $n$ facts at random turn positions, ask about them at turn 20
    - Implement truncation vs. rolling summary; plot fact retention rate vs. turn position for both
    - Target: rolling summary should retain >70% of facts planted before turn 10
- **Part 3 — Break it intentionally**
    - Remove the stopping condition; observe token usage spiral; re-add it
    - Craft a prompt injection in user input; observe effect; implement a sanitisation wrapper
    - Induce sycophancy: prime the model with a false "fact" in its own prior turn, ask it to verify

## Tech Stack

- **LLM**: `transformers` + `unsloth` (GPT OSS 20B 4-bit on colab V100, intentionally with small context to simulate forgetting), otherwise `anthropic` SDK if API credits available (swap in via single config flag at top of notebook)
- `networkx` + `matplotlib` for trajectory visualisation
- Pure Python, no agentic framework this week by design

---

# Week 2 — Memory & RAG

## Goal

- Extend `ChatAgent` with a persistent vector store so it can answer questions over documents that exceed the context window
- Implement and benchmark three RAG variants: vanilla, hybrid, and summary-indexed
- Run an APE (Automatic Prompt Engineering) sweep over the summarization prompt and chunking parameters
- Produce a reusable `RAGStore` module that Week 3 imports directly

## Readings

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — Lewis et al. 2020](https://arxiv.org/abs/2005.11401) the main RAG paper
- [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE) — Gao et al. 2022](https://arxiv.org/abs/2212.10496) Hybrid RAG
- [Large Language Models Are Zero-Shot Reasoners — Kojima et al. 2022](https://arxiv.org/abs/2205.11916) motivation for prompt sensitivity
- [Automatic Prompt Engineer — Zhou et al. 2022](https://arxiv.org/abs/2211.01910) automatic prompt refinement

## Lecture

### Block 1 — Why in-context memory is not enough (15 min)

- **Physical limit**: $|D| \gg L_{\max}$ (e.g. a 500-page PDF vs. a 4K–32K context window)
- Retrieval as a solution: replace $P(y \mid x, \theta)$ with $P(y \mid x, D_{\text{retrieved}}, \theta)$
- The retrieval–generation contract: the LLM can only be as good as what gets retrieved (garbage in, garbage out)

### Block 2 — RAG variants (35 min)

- **Vanilla RAG:** chunk → embed → cosine similarity → top-$k$ → stuff into prompt
    - Chunking strategies: fixed-size, sentence-boundary, semantic — each changes what $\mathbf{e}_{\text{chunk}}$ encodes
    - Chunk size vs. retrieval precision tradeoff
- **Hybrid RAG:** dense + sparse retrieval; BM25 score: $$s_{\text{BM25}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}$$ Fusion: $s = \alpha , s_{\text{dense}} + (1-\alpha) , s_{\text{BM25}}$; $\alpha$ is a tunable hyperparameter
- **Summary-indexed RAG** _(the week's main idea):_
    - Store: `(chunk_id, chunk_text, summary, embedding_of_summary)`
    - Retrieve by $\cos(\mathbf{e}_q, \mathbf{e}_{\text{summary}})$; return `chunk_text`
    - Why it works: the summary strips boilerplate/noise, so $\cos(\mathbf{e}_q, \mathbf{e}_{\text{summary}}) \geq \cos(\mathbf{e}_q, \mathbf{e}_{\text{chunk}})$ tends to hold for question-style queries
    - Related work: RAPTOR (recursive abstractive processing); HyDE (embed a _hypothetical answer_ instead of the query)
- **Multi-turn RAG:** each retrieval is conditioned on prior retrievals; agent reformulates query based on what it got back — this is the Week 1 ReAct loop with the `lookup` tool replaced by the vector store

### Block 3 — Automatic Prompt Engineering (APE) (30 min)

- Motivating question: does the summarisation prompt matter? (Yes, measurably)
- APE (Zhou et al. 2022): treat the prompt as a parameter; let the LLM propose its own instructions
- The optimization objective: $$p^* = \arg\max_{p \in \mathcal{P}} \mathbb{E}_{(x,y) \sim \mathcal{D}}\left[\text{metric}\left(f_p(x),, y\right)\right]$$ where $\mathcal{P}$ is sampled by prompting the LLM to generate candidate instructions
- Connect to DSPy: this is formalized as "signature optimisation" in production systems
- Practical sweep: 3 summarization prompt variants × 2 chunk sizes × 2 values of $k$ → 12-cell grid, runs in ~5 min, visualize the prompting shift over iterations

### Block 4 — Safety & Pitfalls: Week 2 (15 min)

- **Retrieval poisoning:** attacker-controlled document in the store injects instructions into retrieved context
- **Hallucination under retrieval failure:** when top-$k$ returns irrelevant chunks, LLM confabulates rather than abstaining; mitigation: confidence / similarity threshold before stuffing context
- **Chunking artifacts:** a fact split across a chunk boundary is never retrieved correctly; motivate overlap and semantic chunking
- **APE collapse:** if the candidate prompt set is too narrow, optimisation converges to a local minimum; diversity in candidate generation matters

## Example Notebook — `agent_week2.ipynb`

- **Setup:** `from agent_week1 import ChatAgent`
    
- **Exercise 1 — Build the ingestion pipeline**
    - Load a long PDF (100+ pages, either a book or an article)
    - Summarize each chunk and store as embed `(chunk_id, chunk_text, summary, embedding)` in LanceDB
- **Exercise 2 — RAG variants comparison**
    - Implement vanilla, hybrid (LanceDB FTS + vector), and summary-indexed retrieval
    - Hand-label 15 QA pairs from the document (provided in the notebook)
    - Plot: Recall@$k$ for each variant, $k \in {1, 3, 5}$
    - Expected result: summary-indexed outperforms vanilla on question-style queries; hybrid wins on keyword-heavy queries
- **Exercise 3 — APE sweep**
    - Define 3 candidate summarization prompts (e.g., "summarise in 2 sentences", "extract key facts as bullets", "write a dense academic abstract")
    - Re-ingest, re-run the 15-QA benchmark for each
    - Plot accuracy vs. prompt; discuss why one wins
    - _Extension:_ ask the LLM to propose a 4th prompt (the actual APE step); evaluate it and add to the plot (leave this as an optional exercise)
- **Exercise 4 — Plug into `ChatAgent`**
    - Replace the toy `dict` KB from Week 1 with the LanceDB retriever
    - Demo a 20-turn conversation over the ingested document; verify facts planted in the document are retrievable at turn 20

## Tech Stack
- `transformers` + `unsloth` (same LLM as Week 1)
- `lancedb` file-based, native hybrid search; pin version in notebook header
- `sentence-transformers` with `nomic-embed-text-v1.5` (137M, CPU-friendly and fits in GPU alongside)
- `rank_bm25` for sparse retrieval
- `pypdf` for document loading
- `pandas` + `seaborn` for the APE heatmap

---

# Week 3 — Programmatic Tool Calling

## Goal

- Extend the Week 2 agent with a dynamic tool registry supporting Google search, web fetch, and summarise→store
- Understand Anthropic's native tool call protocol at the JSON level — no framework magic
- Introduce programmatic / dynamic tool generation: tools selected at runtime by embedding similarity
- End with a full benchmark: the student's agent from Weeks 1–3 combined, evaluated on fresh web-search questions

## Readings

- [Toolformer: Language Models Can Teach Themselves to Use Tools — Schick et al. 2023](https://arxiv.org/abs/2302.04761) original tool call idea
- [Claude's Documentation of Programmatic Tool Calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) programmatic tool call
- [Constitutional AI — Bai et al. 2022](https://arxiv.org/abs/2212.06950) agentic safety

## Lecture

### Block 1 — Tools as first-class citizens (20 min)

- Toolformer insight: tool calls can be _learned_ as tokens in the sequence. This is how an LLM knows "when" to do a tool call
- Formal model: action space expands to $\mathcal{A} = \mathcal{A}_{\text{text}} \cup {T_1, \ldots, T_n}$ where each $T_i : \mathcal{I}_i \to \mathcal{O}_i$, link the idea to thinking token
- **Reliability degradation in tool chains:** if each tool succeeds with probability $p$, a chain of $n$ tools has end-to-end reliability $p^n$
    - Example: $p = 0.9$, $n = 5 \Rightarrow$ reliability = 59% — motivates robust error handling

### Block 2 — Programmatic / Dynamic Tool Calling (30 min)

- **Motivation**: Hard coded tools are fine for small number of tools, but what about 50+
- **Tool registry pattern:** `dict[str, Callable]` + auto-generated JSON Schema from Pydantic models
- **Tool retrieval:** given a user query, select top-$k$ relevant tools before passing to LLM: $$\text{relevant tools} = \underset{j}{\text{top-}k}; \cos!\left(\mathbf{e}_{\text{query}},; \mathbf{e}_{T_j}\right)$$link this back to RAG since it's the same concept (cosine similarity)
- PydanticAI's `@agent.tool` decorator: show how it auto-generates the schema (production-ready library)
- Parallel vs. sequential tool calls: when the LLM emits multiple `tool_use` blocks in one response; order of execution matters for side-effecting tools

### Block 3 — Implementing real tools (25 min)

- **Tool 1 — Web Search:** simplest, shoot an API call to Google, Brave, or SearXNG
- **Tool 2 — Web Fetch:** `httpx` + `trafilatura` for main-content extraction (strips nav, ads, boilerplate). Discuss the effects of clean text vs raw HTML
- **Tool 3 — Summary and Store:** Week 2's ingestion pipeline being used as a programmatic tool call; expects raw contents and outputs to LanceDB. This prevents model from running out of context on fetches.
- **Tool 4 — Query from store:** query from the indexed DB.
- Full Workflow: user query -> search -> fetch -> summarize -> retrieve top results based on summary DB
- **Number of tool calls / tokens spent as a metric:** start with a weak summarization prompt (from week 2) expecting several loops for the model to arrive at the conclusion after 5+ times, then use an APE'd prompt expecting a reduction in number of calls and tokens spent.

### Block 4 — Safety & Pitfalls: Week 3 (15 min)

- **Prompt injection via web content (critical):** scraped text may contain `"Ignore previous instructions..."`
- **Tool call loops:** search → finds new link → fetches → finds another link → ...; mitigation: visited-URL set + `max_depth` parameter
- **Irreversible actions:** sending emails, posting content, modifying files — the case for a human-in-the-loop confirmation gate before any side-effecting tool
- **Rate limiting & cost:** unconstrained tool calling burns API credits; enforce per-session tool call budgets

## Example Notebook — `agent_week3.ipynb`

- **Setup:** `from agent_week1 import ChatAgent`, `from agent_week2 import RAGStore`
    
- **Exercise - Putting it all together**
    - Implement `class ToolRegistry` with `register(name, fn, schema)`, `get_schemas()`, `call(name, args)` methods
    - Define Pydantic models for each tool's input; auto-generate JSON Schema
    - Register the 4 tools: `web_search`, `web_fetch`, `summary_and_store`, `query_store` as in lecture
    - Test - ask the model to search for *latest* publications in arXiv
    - Leave a blank notebook cell to encourage defining more custom tools

## Tech Stack

- `pydantic-ai` for tool registry and schema generation
- `lancedb` + `sentence-transformers` (inherited from Week 2)
- `httpx` for web fetch and `trafilatura` for content extraction
- `transformers` + `unsloth` (same LLM throughout); `anthropic` SDK if API credits available

---

## Cross-Week Benchmark Summary

| Week | Dataset                                                                                                    | Metric                                                   | Runtime target |
| ---- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | -------------- |
| 1    | Synthetic: 20-turn conversation with planted facts                                                         | Fact retention rate @ turn 15+ (answer accuracy/F1)      | < 1 min        |
| 2    | 15 hand-labelled QA pairs over one long PDF (`natural_questions` or `qasper` filtered to docs > 4K tokens) | Recall@k ($k \in {1,3,5}$), answer accuracy/F1           | < 2 min        |
| 3    | 10 fresh ArXiv questions generated morning-of (generator script provided)                                  | Answer accuracy, numbers of tool calls and tokens spent. | < 3 min        |

## Final Project Template (end of Week 3)

The student's combined notebook will have the following

```
ChatAgent (Week 1)
  └── in-context memory with rolling summary compression
  └── ReAct loop with max_iterations guard
RAGStore (Week 2)
  └── LanceDB vector store (summary-indexed)
  └── Hybrid retrieval (dense + BM25)
  └── APE-optimised summarisation prompt
ToolRegistry (Week 3)
  └── web_search
  └── web_fetch (with sanitization optionally?)
  └── summary_and_store
  └── query_store (RAG from week 2)
  └── Embedding-based top-k tool selection
```