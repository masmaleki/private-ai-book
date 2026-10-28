# Operational AI Engineering Toolkit, Open-Source Repositories & Companion Code Repository

!!! goal "In This Appendix"
    - You will find the URL and directory structure of **this book's public GitHub repository (`github.com/masmaleki/private-ai-book`)** so you can clone and run all Python, Node.js, and PHP 8.3 code examples in one place.
    - You will have a **curated, layer-by-layer reference catalog of the best open-source AI engineering repositories** as of 2026—spanning GPU inference serving, LoRA/QLoRA fine-tuning, Hybrid RAG, the Model Context Protocol (`MCP`), multi-agent orchestration, evaluation, and classical/multilingual NLP.
    - For every tool, you will see **what problem it solves**, **which chapter of this book teaches its underlying concepts**, and **which programming languages (`Python`, `TypeScript / Node.js`, `PHP / Laravel`, `C++ / Go / Rust`)** it supports.

## 1. Official Companion Code Repository & Live Toolkit Index on GitHub

Every runnable script in this book, along with the automated test suites (`pytest`), multi-language reference implementations in **Python**, **Node.js (TypeScript / ESM)**, and **PHP 8.3 (Laravel)**, and a live, continuously updated version of the tables in this appendix, is publicly available at:

- **Official Companion GitHub Repository:**  
  [`https://github.com/masmaleki/private-ai-book`](https://github.com/masmaleki/private-ai-book)

To clone and run the verification suites on your own machine:

```bash
git clone https://github.com/masmaleki/private-ai-book.git
cd private-ai-book

# Run tests for the Hybrid NLP Pipeline (Ch. 12) and Agentic AI Architecture (Ch. 34)
pytest code/ch12/ code/ch34/ -v

# Run the Secure Web Research Subagent & 4-Tier Memory Architecture (Ch. 34)
python3 code/ch34/web_research_memory_agent.py

# Run the Multi-Agent Subagent Orchestrator in Node.js
node code/ch34/agentic_node.mjs
```

Because the open-source AI ecosystem evolves rapidly, whenever a major new library or version of the tools covered in this appendix is released, the `README.md` catalog in the repository above is updated accordingly.

---

## 2. Layer 1: Private LLM Serving & Inference Engines

When you want to deploy an open-weight model (such as the `Qwen`, `Llama`, `Mistral`, or `Gemma` families) on your own internal servers (Chapters 21 and 41) and expose a standard OpenAI-compatible API to your backend layer, these five repositories form your industrial backbone:

| Tool / Repository | GitHub URL | Primary Use Case & Core Strength | Reference Chapters |
|---|---|---|---|
| **vLLM** | [`github.com/vllm-project/vllm`](https://github.com/vllm-project/vllm) | The gold standard for high-throughput GPU LLM serving; features **PagedAttention**, **Continuous Batching**, concurrent multi-**LoRA Adapter** serving, and structured outputs. | 18, 21, 37 |
| **llama.cpp** | [`github.com/ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) | Ultra-fast C/C++ inference engine with zero heavy dependencies; ideal for **GGUF** models, hybrid CPU+GPU execution, workstations, edge servers, and air-gapped deployments. | 19, 21, 41 |
| **SGLang** | [`github.com/sgl-project/sglang`](https://github.com/sgl-project/sglang) | High-speed serving engine featuring **RadixAttention** (automatic KV-cache reuse across shared prompt prefixes) and near-zero-overhead constrained JSON decoding; ideal for multi-agent systems. | 18, 21, 22 |
| **Text Generation Inference (TGI)** | [`github.com/huggingface/text-generation-inference`](https://github.com/huggingface/text-generation-inference) | Hugging Face's official Rust/Python inference server for stable containerized production deployments. | 21 |
| **Ollama** | [`github.com/ollama/ollama`](https://github.com/ollama/ollama) | The simplest developer tool (built in Go on top of `llama.cpp`) for spinning up local models with a single CLI command during prototyping. | 20, 21 |

Table E-1: Reference open-source repositories for private LLM serving and inference.

---

## 3. Layer 2: Training, Fine-Tuning, LoRA/QLoRA, and Alignment

When the decision tree in Chapter 35 confirms that your model needs to learn a new **behavior, output format, or domain tone**, these libraries (Chapters 36 and 37) dramatically reduce GPU memory requirements and training time:

| Tool / Repository | GitHub URL | Primary Use Case & Core Strength | Reference Chapters |
|---|---|---|---|
| **Unsloth** | [`github.com/unslothai/unsloth`](https://github.com/unslothai/unsloth) | Custom **OpenAI Triton** kernels and manual backpropagation engine; delivers **2x to 5x faster training** and **up to 70% less VRAM usage** for LoRA, QLoRA, and GRPO with `0%` accuracy loss and direct export to `vLLM` and `GGUF`. | 37 |
| **Hugging Face PEFT** | [`github.com/huggingface/peft`](https://github.com/huggingface/peft) | The reference parameter-efficient fine-tuning library (LoRA, DoRA, QLoRA, IA3) tightly integrated across the Hugging Face Transformers ecosystem. | 37 |
| **Hugging Face TRL** | [`github.com/huggingface/trl`](https://github.com/huggingface/trl) | Standard post-training library providing `SFTTrainer`, preference alignment (`DPOTrainer`), and reinforcement learning (`GRPOTrainer`). | 17, 37 |
| **Axolotl** | [`github.com/axolotl-ai-cloud/axolotl`](https://github.com/axolotl-ai-cloud/axolotl) | YAML-driven fine-tuning framework; ideal for teams running multi-GPU distributed training (DeepSpeed / FSDP) without writing boilerplate code. | 37 |
| **LLaMA-Factory** | [`github.com/hiyouga/LLaMA-Factory`](https://github.com/hiyouga/LLaMA-Factory) | Comprehensive fine-tuning suite supporting 100+ open models with both an interactive WebUI (for rapid experiments) and a production CLI. | 37 |
| **bitsandbytes** | [`github.com/bitsandbytes-foundation/bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) | Foundational library for 4-bit (`NF4`) and 8-bit quantization and paged optimizers enabling QLoRA on consumer and workstation GPUs. | 19, 37 |

Table E-2: Reference open-source repositories for fine-tuning and memory-efficient training.

---

## 4. Layer 3: Agentic AI, MCP, and Multi-Language Orchestration

As we saw in Chapters 23, 33, and 34, the agentic layer (Brain + Hands + MCP Nervous System + Specialist Subagents) is never restricted to Python—you can build it natively in your primary ERP backend language:

| Tool / Repository | GitHub URL | Languages | Primary Use Case & Core Strength | Reference Chapters |
|---|---|---|---|---|
| **Model Context Protocol (MCP)** | [`github.com/modelcontextprotocol`](https://github.com/modelcontextprotocol) | Python, TS, C#, Go, Java | Official specification and SDKs for **MCP** (the nervous system of the modern AI stack) to build standardized tool servers and clients. | 23, 34 |
| **PydanticAI** | [`github.com/pydantic/pydantic-ai`](https://github.com/pydantic/pydantic-ai) | Python | Type-safe agent framework with strict schema validation, dependency injection, and structured outputs built by the creators of Pydantic. | 22, 33, 34 |
| **LangGraph** | [`github.com/langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph) | Python, TS | Builds agentic and multi-agent workflows as **stateful graphs** with first-class support for Human-in-the-Loop gates, persistence, and Reflection loops. | 33, 34 |
| **OpenAI Agents SDK** | [`github.com/openai/openai-agents-python`](https://github.com/openai/openai-agents-python) | Python | Lightweight, minimal framework for defining agents, subagent **handoffs**, guardrails, and tracing; works seamlessly with any OpenAI-compatible server such as `vLLM`. | 33, 34 |
| **Vercel AI SDK** | [`github.com/vercel/ai`](https://github.com/vercel/ai) | TypeScript / Node.js | The premier AI engineering toolkit for **Node.js and TypeScript**; provides unified Zod structured outputs, tool calling, multi-step agent loops, and private model adapters. | 22, 33, 34 |
| **Prism (Laravel AI)** | [`github.com/echolabsdev/prism`](https://github.com/echolabsdev/prism) | PHP / Laravel | Idiomatic **Laravel** package for working with LLMs, defining tools, enforcing structured outputs, and connecting to private or cloud inference servers. | 22, 33, 34 |
| **OpenAI PHP Client** | [`github.com/openai-php/client`](https://github.com/openai-php/client) | PHP / Laravel | Standard zero-dependency PHP client that connects directly to internal `vLLM` or `Ollama` servers by setting a custom `baseUri`. | 21, 34 |

Table E-3: Reference repositories for Agentic AI and MCP across Python, Node.js, and PHP.

---

## 5. Layer 4: RAG, Document Parsing, Vector Search, and Analytical Engines

In Chapters 24, 25, and 28, we saw that answer quality depends directly on clean PDF/table extraction and hybrid retrieval combined with SQL analytics:

| Tool / Repository | GitHub URL | Primary Use Case & Core Strength | Reference Chapters |
|---|---|---|---|
| **Docling** | [`github.com/docling-project/docling`](https://github.com/docling-project/docling) | Outstanding open-source document parser (originated at IBM Research) that converts complex PDFs, datasheets, and engineering tables into clean Markdown and JSON ready for RAG chunking. | 24, 25 |
| **Qdrant** | [`github.com/qdrant/qdrant`](https://github.com/qdrant/qdrant) | High-performance Rust vector and hybrid (Sparse + Dense) search engine with rich payload filtering for strict multi-tenant isolation (`tenant_id`). | 14, 24, 25 |
| **ClickHouse** | [`github.com/ClickHouse/ClickHouse`](https://github.com/ClickHouse/ClickHouse) | Ultra-fast columnar OLAP database for supply-chain analytics, price percentile computation (`P10/P50/P90`), audit logs, and LLM telemetry storage. | 1, 23, 28, 44 |
| **chDB** | [`github.com/chdb-io/chdb`](https://github.com/chdb-io/chdb) | In-process ClickHouse SQL engine for Python and Node.js; queries Parquet, CSV, and JSON files directly at columnar speed with zero server setup. | 6, 28 |
| **Sentence-Transformers** | [`github.com/UKPLab/sentence-transformers`](https://github.com/UKPLab/sentence-transformers) | Standard library for running local **Embedding** models (such as `bge-m3` and `multilingual-e5`) and **Cross-Encoder Rerankers** for multilingual retrieval. | 12, 14, 24, 25 |
| **FlagEmbedding (BGE)** | [`github.com/FlagOpen/FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding) | Home of `BGE-M3` (simultaneous Dense, Sparse/Lexical, and Multi-Vector retrieval across 100+ languages including English, German, and Persian) and precision rerankers. | 14, 24, 25 |

Table E-4: Reference repositories for RAG, document parsing, vector search, and analytical databases.

---

## 6. Layer 5: Classical NLP, Multilingual Processing, and Tabular Machine Learning

Before waking up a GPU-heavy LLM (Chapter 12), and whenever you need calibrated numerical or time-series forecasts (Chapters 7–10 and 28), these CPU-native libraries are unmatched:

| Tool / Repository | GitHub URL | Primary Use Case & Core Strength | Reference Chapters |
|---|---|---|---|
| **spaCy** | [`github.com/explosion/spaCy`](https://github.com/explosion/spaCy) | Industrial-strength Python/Cython NLP library for fast tokenization, POS tagging, dependency parsing, and custom CPU-based NER pipelines. | 12 |
| **RapidFuzz** | [`github.com/rapidfuzz/RapidFuzz`](https://github.com/rapidfuzz/RapidFuzz) | Ultra-fast C++ implementation of **Levenshtein**, **Jaro-Winkler**, and fuzzy string matching; ideal for normalizing part numbers (MPNs) and supplier names at million-row scale. | 12, 26 |
| **Hazm** | [`github.com/roshan-research/hazm`](https://github.com/roshan-research/hazm) | Specialized Persian NLP library for Unicode/ZWNJ normalization, stemming, lemmatization, and POS tagging in multilingual datasets. | 12 |
| **ParsBERT / EuroBERT** | [`github.com/hooshvare/parsbert`](https://github.com/hooshvare/parsbert) | Pre-trained **Encoder-Only** Transformer models for sub-15 ms text classification, sentiment analysis, and Named Entity Recognition (NER). | 12 |
| **Hugging Face Tokenizers** | [`github.com/huggingface/tokenizers`](https://github.com/huggingface/tokenizers) | Rust implementation of BPE, WordPiece, and Unigram tokenizers for ultra-fast text preprocessing. | 13 |
| **scikit-learn** | [`github.com/scikit-learn/scikit-learn`](https://github.com/scikit-learn/scikit-learn) | The standard toolkit for classical machine learning (regression, decision trees, Random Forests, TF-IDF, Naive Bayes, clustering, and evaluation metrics). | 6–10, 12 |
| **LightGBM** | [`github.com/microsoft/LightGBM`](https://github.com/microsoft/LightGBM) | High-speed Gradient Boosted Decision Trees engine; our primary choice for tabular price prediction, quantile regression (`P10/P50/P90`), and lead-time forecasting. | 8, 28, 29 |

Table E-5: Reference repositories for classical/multilingual NLP and tabular machine learning.

---

## 7. Layer 6: Evaluation, Security, Red-Teaming, and Observability

No enterprise AI system should ever reach production without automated evaluation gates (Chapters 38 and 39), adversarial red-teaming (Chapter 40), and production tracing (Chapter 44):

| Tool / Repository | GitHub URL | Primary Use Case & Core Strength | Reference Chapters |
|---|---|---|---|
| **Promptfoo** | [`github.com/promptfoo/promptfoo`](https://github.com/promptfoo/promptfoo) | Essential CLI and CI/CD tool for prompt regression testing, side-by-side model comparison, and automated **Red-Teaming** (prompt injection, data leakage, and jailbreak probes). | 38, 39, 40 |
| **Langfuse** | [`github.com/langfuse/langfuse`](https://github.com/langfuse/langfuse) | Open-source, **self-hostable** LLM observability platform for step-by-step subagent trace logging, prompt version management, and latency/cost telemetry. | 34, 38, 44 |
| **Arize Phoenix** | [`github.com/Arize-ai/phoenix`](https://github.com/Arize-ai/phoenix) | OpenTelemetry-based open-source tracing and evaluation platform for LLMs, RAG pipelines, and multi-agent workflows. | 38, 44 |
| **Ragas** | [`github.com/explodinggradients/ragas`](https://github.com/explodinggradients/ragas) | Specialized evaluation framework for **RAG** pipelines (measuring answer `Faithfulness`, `Context Precision`, and retrieval recall). | 25, 38 |
| **Microsoft Presidio** | [`github.com/microsoft/presidio`](https://github.com/microsoft/presidio) | Open-source engine for detecting, redacting, and pseudonymizing **PII** in text and logs prior to storage or fine-tuning. | 36, 40, 42 |
| **LLM Guard** | [`github.com/protectai/llm-guard`](https://github.com/protectai/llm-guard) | Fast security scanners for LLM inputs and outputs (detecting prompt injection, secret leakage, PII exposure, and schema violations). | 40 |

Table E-6: Reference repositories for evaluation, security guardrails, and production observability.

---

## 8. At-a-Glance Tool Selection Blueprint for Your Next Project

Whenever you sit in an architecture review meeting and need to choose the right open-source tool for each layer of your stack, use this seven-step blueprint:

1. **For initial triage, text normalization, and MPN fuzzy matching on CPU:** Combine `Regex` + `spaCy / Hazm` + `RapidFuzz` + `scikit-learn / BERT Encoder` (Chapter 12).
2. **For numerical price, demand, and lead-time forecasting on tabular ERP data:** Run `LightGBM` + `scikit-learn` on top of `ClickHouse` / `MariaDB` (Chapters 8 and 28).
3. **For private LLM serving on internal GPU servers:** Use `vLLM` (or `SGLang`) for high-throughput production serving, and `llama.cpp` for lightweight, edge, or air-gapped environments (Chapter 21).
4. **For document and datasheet search (Hybrid RAG):** Parse PDFs with `Docling` + embed and rerank with `BGE-M3` + index in `Qdrant` or `Elasticsearch` with mandatory `tenant_id` filtering (Chapters 24 and 25).
5. **For building Agents, loading on-demand `SKILL.md` playbooks, and running isolated Subagents:** In **PHP / Laravel**, use `Http::pool()` and `Prism`; in **Node.js / TypeScript**, use the `Vercel AI SDK` and `@modelcontextprotocol/sdk`; and in **Python**, use `PydanticAI` or `LangGraph`—backed by `OutboundQuerySanitizer` and the 4-Tier Memory Architecture for external research (Chapters 33 and 34).
6. **For fast, low-VRAM fine-tuning of a behavior or output format:** Curate and audit your dataset with the Chapter 36 validators + train LoRA/QLoRA adapters with **`Unsloth`** + serve multi-LoRA adapters on `vLLM` (Chapter 37).
7. **For release gates and production monitoring:** Run automated evals and red-teaming with `Promptfoo` in CI/CD + self-host `Langfuse` for full trace observability (Chapters 38–44).
