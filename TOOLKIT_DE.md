# Operativer KI-Engineering-Werkzeugkasten, Open-Source-Repositories & Code-Repository zum Buch

!!! goal "In diesem Anhang"
    - Du findest die Adresse und Struktur des **öffentlichen GitHub-Repositorys zu diesem Buch (`github.com/masmaleki/private-ai-book`)**, damit du sämtliche Codebeispiele in Python, Node.js und PHP 8.3 an einem Ort klonen und ausführen kannst.
    - Du erhältst einen **kuratierten, nach Schichten geordneten Referenzkatalog der besten Open-Source-Repositories der KI-Welt** im Jahr 2026 — vom GPU-Serving und LoRA-/QLoRA-Fine-Tuning über Hybrid-RAG, das Model Context Protocol (`MCP`) und die Multi-Agenten-Orchestrierung bis hin zu Evaluierung und klassischem/mehrsprachigem NLP.
    - Für jedes Werkzeug siehst du sofort, **welches Problem es löst**, **in welchem Kapitel dieses Buches du die zugrundeliegenden Konzepte gelernt hast** und **in welchen Programmiersprachen (`Python`, `TypeScript / Node.js`, `PHP / Laravel`, `C++ / Go / Rust`)** es einsetzbar ist.

## 1. Offizielles Code-Repository & lebendiger Werkzeugkatalog auf GitHub

Alle ausführbaren Skripte dieses Buches, die automatisierten Test-Suites (`pytest`), die mehrsprachigen Referenzimplementierungen in **Python**, **Node.js (TypeScript / ESM)** und **PHP 8.3 (Laravel)** sowie eine laufend aktualisierte Fassung der Tabellen dieses Anhangs stehen dir öffentlich unter folgender Adresse zur Verfügung:

- **GitHub-Repository zum Buch:**  
  [`https://github.com/masmaleki/private-ai-book`](https://github.com/masmaleki/private-ai-book)

So klonst und startest du die Beispiele direkt auf deinem System:

```bash
git clone https://github.com/masmaleki/private-ai-book.git
cd private-ai-book

# Tests für die hybride NLP-Pipeline (Kap. 12) und Agentic-AI-Architektur (Kap. 34) ausführen
pytest code/ch12/ code/ch34/ -v

# Sicheren Web-Research-Subagenten & 4-stufige Gedächtnisarchitektur starten (Kap. 34)
python3 code/ch34/web_research_memory_agent.py

# Multi-Agenten-Subagent-Orchestrator in Node.js ausführen
node code/ch34/agentic_node.mjs
```

Da sich die Welt der Open-Source-KI-Werkzeuge jeden Monat weiterentwickelt, wird die Tabelle in der `README.md` des oben genannten Repositorys laufend aktualisiert, sobald wichtige neue Bibliotheken oder neue Versionen erscheinen.

---

## 2. Schicht 1: Privates Serving und Betrieb von Sprachmodellen (Inference & Model Serving)

Wenn du ein Open-Weight-Modell (wie die Familien `Qwen`, `Llama`, `Mistral` oder `Gemma`) auf deinen eigenen internen Servern (Kapitel 21 und 41) betreiben und deinem Backend eine standardisierte, OpenAI-kompatible API bereitstellen möchtest, bilden diese fünf Repositories dein industrielles Rückgrat:

| Werkzeug / Repository | GitHub-Adresse | Haupteinsatzzweck & besondere Stärke | Referenzkapitel |
|---|---|---|---|
| **vLLM** | [`github.com/vllm-project/vllm`](https://github.com/vllm-project/vllm) | Der Goldstandard für industrielles LLM-Serving auf GPUs; ausgestattet mit **PagedAttention**, **Continuous Batching**, gleichzeitigem Serving mehrerer **LoRA-Adapter** und strukturierten Ausgaben. | 18, 21, 37 |
| **llama.cpp** | [`github.com/ggml-org/llama.cpp`](https://github.com/ggml-project/llama.cpp) | Ultraschnelle C/C++-Engine ohne schwere Abhängigkeiten; ideal für das **GGUF**-Format, hybriden CPU+GPU-Betrieb, Workstations, schlanke Server und Air-gapped-Umgebungen. | 19, 21, 41 |
| **SGLang** | [`github.com/sgl-project/sglang`](https://github.com/sgl-project/sglang) | Hochgeschwindigkeits-Serving-Engine mit **RadixAttention** (automatisches KV-Caching gemeinsamer Prompt-Präfixe) und nahezu verlustfreiem Constrained JSON Decoding; ideal für Multi-Agenten-Systeme. | 18, 21, 22 |
| **Text Generation Inference (TGI)** | [`github.com/huggingface/text-generation-inference`](https://github.com/huggingface/text-generation-inference) | Offizieller Inferenzserver von Hugging Face mit Rust-/Python-Kern für den stabilen Container-Betrieb. | 21 |
| **Ollama** | [`github.com/ollama/ollama`](https://github.com/ollama/ollama) | Das einfachste Werkzeug (auf Basis von Go und `llama.cpp`), um Modelle in der Prototyping-Phase mit einem einzigen Befehl lokal auf Entwickler-Laptops zu starten. | 20, 21 |

Tabelle E-1: Referenz-Repositories für privates LLM-Serving und Inferenz.

---

## 3. Schicht 2: Training, Fine-Tuning, LoRA/QLoRA und Alignment

Wenn du anhand des Entscheidungsbaums aus Kapitel 35 feststellst, dass dein Modell ein neues **Verhalten, Ausgabeformat oder einen bestimmten Tonfall** lernen soll, senken diese Werkzeuge (Kapitel 36 und 37) die GPU-Kosten und die Trainingszeit drastisch:

| Werkzeug / Repository | GitHub-Adresse | Haupteinsatzzweck & besondere Stärke | Referenzkapitel |
|---|---|---|---|
| **Unsloth** | [`github.com/unslothai/unsloth`](https://github.com/unslothai/unsloth) | Maßgeschneiderte **OpenAI-Triton**-Kernel und manuelle Gradienten-Ableitung; **2- bis 5-mal schnelleres Training** und **bis zu 70 % weniger VRAM-Verbrauch** bei LoRA, QLoRA und GRPO ohne Genauigkeitsverlust sowie direkter Export für `vLLM` und `GGUF`. | 37 |
| **Hugging Face PEFT** | [`github.com/huggingface/peft`](https://github.com/huggingface/peft) | Die Referenzbibliothek für parametereffiziente Verfahren (LoRA, DoRA, QLoRA, IA3), nahtlos integriert in das gesamte Transformers-Ökosystem. | 37 |
| **Hugging Face TRL** | [`github.com/huggingface/trl`](https://github.com/huggingface/trl) | Standard-Trainer für das Nachtraining inklusive `SFTTrainer`, Präferenz-Alignment (`DPOTrainer`) und Reasoning-Training (`GRPOTrainer`). | 17, 37 |
| **Axolotl** | [`github.com/axolotl-ai-cloud/axolotl`](https://github.com/axolotl-ai-cloud/axolotl) | YAML-gestütztes Fine-Tuning-Framework; ideal für Teams, die auf mehreren GPUs (mit DeepSpeed und FSDP) ohne wiederkehrenden Boilerplate-Code trainieren möchten. | 37 |
| **LLaMA-Factory** | [`github.com/hiyouga/LLaMA-Factory`](https://github.com/hiyouga/LLaMA-Factory) | Umfassendes Fine-Tuning-Framework für über 100 offene Modelle mit grafischer WebUI (für schnelle Experimente) und produktionsreifer CLI. | 37 |
| **bitsandbytes** | [`github.com/bitsandbytes-foundation/bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) | Grundlegende Bibliothek für 4-Bit- (`NF4`) und 8-Bit-Quantisierung sowie Paged Optimizer für QLoRA auf Consumer- und Workstation-Grafikkarten. | 19, 37 |

Tabelle E-2: Referenz-Repositories für Fine-Tuning und speichereffizientes Training.

---

## 4. Schicht 3: Agentic AI, MCP und mehrsprachige Orchestrierung

Wie wir in den Kapiteln 23, 33 und 34 gesehen haben, ist die Agenten-Schicht (Gehirn + Hände + MCP-Nervensystem + Spezialisten-Subagents) keineswegs auf Python beschränkt — du kannst sie direkt in der Hauptsprache deines ERP-Backends entwickeln:

| Werkzeug / Repository | GitHub-Adresse | Sprachen | Haupteinsatzzweck & besondere Stärke | Referenzkapitel |
|---|---|---|---|---|
| **Model Context Protocol (MCP)** | [`github.com/modelcontextprotocol`](https://github.com/modelcontextprotocol) | Python, TS, C#, Go, Java | Offizielles Repository des **MCP**-Standards (das Nervensystem des KI-Stacks) inklusive offizieller SDKs für Tool-Server und -Clients. | 23, 34 |
| **PydanticAI** | [`github.com/pydantic/pydantic-ai`](https://github.com/pydantic/pydantic-ai) | Python | Typsicheres Agenten-Framework mit strikter Datenvalidierung, Dependency Injection und strukturierten Ausgaben vom Team hinter Pydantic. | 22, 33, 34 |
| **LangGraph** | [`github.com/langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph) | Python, TS | Aufbau von Agenten- und Multi-Agenten-Systemen als **zustandsbehafteter Graph (Stateful Graph)** mit erstklassiger Unterstützung für Human-in-the-Loop-Gates und Reflection. | 33, 34 |
| **OpenAI Agents SDK** | [`github.com/openai/openai-agents-python`](https://github.com/openai/openai-agents-python) | Python | Schlankes, minimalistisches Framework zur Definition von Agenten, Aufgabenübergaben an Subagenten (**Handoffs / Subagents**), Guardrails und Tracing; kompatibel mit jedem `vLLM`-Server. | 33, 34 |
| **Vercel AI SDK** | [`github.com/vercel/ai`](https://github.com/vercel/ai) | TypeScript / Node.js | Der beliebteste KI-Engineering-Baukasten in der **Node.js- und TypeScript**-Welt; durchgängige Unterstützung für strukturierte Zod-Ausgaben, Tool Calling, Agent-Schleifen und private Modelle. | 22, 33, 34 |
| **Prism (Laravel AI)** | [`github.com/echolabsdev/prism`](https://github.com/echolabsdev/prism) | PHP / Laravel | Natives, modernes **Laravel**-Paket für die Arbeit mit LLMs, Tool-Definitionen, Structured Outputs und die Anbindung privater oder Cloud-basierter Inferenzserver. | 22, 33, 34 |
| **OpenAI PHP Client** | [`github.com/openai-php/client`](https://github.com/openai-php/client) | PHP / Laravel | Standardisierter, abhängigkeitsfreier PHP-Client, der sich durch Anpassung der `baseUri` direkt mit internen `vLLM`- oder `Ollama`-Servern verbindet. | 21, 34 |

Tabelle E-3: Referenz-Repositories für Agentic AI und MCP in Python, Node.js und PHP.

---

## 5. Schicht 4: RAG, Dokumentenextraktion, Vektor- und Analysedatenbanken

In den Kapiteln 24, 25 und 28 haben wir gesehen, dass die Antwortqualität unmittelbar von einer sauberen PDF-/Tabellenextraktion und der Kombination aus hybrider Suche und numerischer Analyse abhängt:

| Werkzeug / Repository | GitHub-Adresse | Haupteinsatzzweck & besondere Stärke | Referenzkapitel |
|---|---|---|---|
| **Docling** | [`github.com/docling-project/docling`](https://github.com/docling-project/docling) | Herausragendes Open-Source-Werkzeug (entwickelt bei IBM Research) zur Umwandlung komplexer PDFs, Datenblätter und technischer Tabellen in sauberes Markdown und JSON für das RAG-Chunking. | 24, 25 |
| **Qdrant** | [`github.com/qdrant/qdrant`](https://github.com/qdrant/qdrant) | In Rust geschriebene Vektor- und Hybrid-Suchmaschine (Sparse + Dense); extrem schnell, speichereffizient und mit präzisem Payload-Filtering für strikte Mandantentrennung (`tenant_id`). | 14, 24, 25 |
| **ClickHouse** | [`github.com/ClickHouse/ClickHouse`](https://github.com/ClickHouse/ClickHouse) | Ultraschnelle spaltenorientierte Analysedatenbank für Lieferketten-Abfragen, Preis-Perzentile (`P10/P50/P90`), Audit-Logs und LLM-Telemetrie. | 1, 23, 28, 44 |
| **chDB** | [`github.com/chdb-io/chdb`](https://github.com/chdb-io/chdb) | In-Process-SQL-Engine von ClickHouse für Python und Node.js; führt schwere analytische Abfragen direkt auf Parquet-, CSV- und JSON-Dateien ohne eigenen Server aus. | 6, 28 |
| **Sentence-Transformers** | [`github.com/UKPLab/sentence-transformers`](https://github.com/UKPLab/sentence-transformers) | Standardbibliothek für die lokale Ausführung von **Embedding**-Modellen (wie `bge-m3` und `multilingual-e5`) sowie **Cross-Encoder-Rerankern** für mehrsprachige Suche. | 12, 14, 24, 25 |
| **FlagEmbedding (BGE)** | [`github.com/FlagOpen/FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding) | Heimat der `BGE-M3`-Modelle (gleichzeitige Unterstützung von Dense, Sparse/Lexical und Multi-Vector in über 100 Sprachen inklusive Deutsch, Englisch und Persisch) sowie präziser Reranker. | 14, 24, 25 |

Tabelle E-4: Referenz-Repositories für RAG, Dokumentenextraktion und Datenbank-Engines.

---

## 6. Schicht 5: Klassisches & mehrsprachiges NLP sowie tabellarisches Machine Learning

Bevor wir ein schweres LLM auf der GPU aufwecken (Kapitel 12) und wann immer wir kalibrierte numerische Vorhersagen oder Zeitreihenprognosen benötigen (Kapitel 7 bis 10 und 28), sind diese Werkzeuge auf der CPU unschlagbar:

| Werkzeug / Repository | GitHub-Adresse | Haupteinsatzzweck & besondere Stärke | Referenzkapitel |
|---|---|---|---|
| **spaCy** | [`github.com/explosion/spaCy`](https://github.com/explosion/spaCy) | Industrielle, extrem schnelle NLP-Bibliothek in Python/Cython für Tokenisierung, POS-Tagging, Dependenzanalyse und maßgeschneiderte NER-Pipelines auf der CPU. | 12 |
| **RapidFuzz** | [`github.com/rapidfuzz/RapidFuzz`](https://github.com/rapidfuzz/RapidFuzz) | Ultraschnelle C++-Implementierung der **Levenshtein**- und **Jaro-Winkler**-Editierdistanz sowie unscharfer String-Suche; ideal zur Bereinigung von Bauteilnummern (MPNs) und Firmennamen im Millionenmaßstab. | 12, 26 |
| **Hazm** | [`github.com/roshan-research/hazm`](https://github.com/roshan-research/hazm) | Spezialisierte Bibliothek für mehrsprachige/persische Textnormalisierung, Halbzeichen-Korrektur (ZWNJ), Stemming und Lemmatisierung. | 12 |
| **EuroBERT / ParsBERT** | [`github.com/hooshvare/parsbert`](https://github.com/hooshvare/parsbert) | Vortrainierte **Encoder-Only**-Transformer-Modelle für Textklassifikation, Sentiment-Analyse und Named Entity Recognition (NER) in unter 15 ms. | 12 |
| **Hugging Face Tokenizers** | [`github.com/huggingface/tokenizers`](https://github.com/huggingface/tokenizers) | Rust-Implementierung der Algorithmen BPE, WordPiece und Unigram für den Bau und Betrieb ultraschneller Tokenizer. | 13 |
| **scikit-learn** | [`github.com/scikit-learn/scikit-learn`](https://github.com/scikit-learn/scikit-learn) | Der Standard-Werkzeugkasten für klassisches Machine Learning (Regression, Entscheidungsbäume, Random Forest, TF-IDF, Naive Bayes, Clustering und Evaluierungsmetriken). | 6 bis 10, 12 |
| **LightGBM** | [`github.com/microsoft/LightGBM`](https://github.com/microsoft/LightGBM) | Ultraschnelle Engine für Gradient Boosted Decision Trees; erste Wahl in der Industrie für Preisprognosen, Quantilsregression (`P10/P50/P90`) und Lieferzeit-Vorhersagen auf tabellarischen ERP-Daten. | 8, 28, 29 |

Tabelle E-5: Referenz-Repositories für klassisches/mehrsprachiges NLP und tabellarisches Machine Learning.

---

## 7. Schicht 6: Evaluierung, Sicherheit, Red-Teaming und Observability

Kein KI-System darf ohne automatisiertes Evaluierungs-Gate (Kapitel 38 und 39), Penetrationstests (Kapitel 40) und Produktions-Tracing (Kapitel 44) in den Produktivbetrieb gehen:

| Werkzeug / Repository | GitHub-Adresse | Haupteinsatzzweck & besondere Stärke | Referenzkapitel |
|---|---|---|---|
| **Promptfoo** | [`github.com/promptfoo/promptfoo`](https://github.com/promptfoo/promptfoo) | Herausragendes CLI-Werkzeug für Prompt-Regressionstests, Modellvergleiche in CI/CD und automatisiertes **Red-Teaming** (Prompt Injection, Datenabfluss und Jailbreaks). | 38, 39, 40 |
| **Langfuse** | [`github.com/langfuse/langfuse`](https://github.com/langfuse/langfuse) | Open-Source-Plattform (vollständig **Self-hosted** betreibbar) für Observability, schrittweises Subagent-Tracing, Prompt-Versionierung sowie Kosten- und Latenzüberwachung. | 34, 38, 44 |
| **Arize Phoenix** | [`github.com/Arize-ai/phoenix`](https://github.com/Arize-ai/phoenix) | Auf OpenTelemetry basierende Open-Source-Plattform für das Tracing und die Evaluierung von LLMs, RAG-Systemen und Agenten. | 38, 44 |
| **Ragas** | [`github.com/explodinggradients/ragas`](https://github.com/explodinggradients/ragas) | Spezialisiertes Evaluierungs-Framework für **RAG**-Pipelines (Messung von Quellentreue `Faithfulness`, `Context Precision` und Antwortabdeckung). | 25, 38 |
| **Microsoft Presidio** | [`github.com/microsoft/presidio`](https://github.com/microsoft/presidio) | Open-Source-Engine zur Erkennung und Pseudonymisierung personenbezogener Daten (**PII Redaction & Pseudonymization**) in Texten und Logs vor der Speicherung oder dem Fine-Tuning. | 36, 40, 42 |
| **LLM Guard** | [`github.com/protectai/llm-guard`](https://github.com/protectai/llm-guard) | Sammlung schneller Sicherheits-Scanner für Ein- und Ausgaben von LLMs (Erkennung von Prompt Injection, Geheimnis-Leaks, PII-Schwärzung und Ausgabevalidierung). | 40 |

Tabelle E-6: Referenz-Repositories für Evaluierung, Sicherheit und Observability.

---

## 8. Auf einen Blick: Der Werkzeug-Entscheidungsfahrplan für dein nächstes Projekt

Immer wenn du in einer Architektursitzung entscheiden musst, welches Open-Source-Repository du für welche Schicht auswählst, nutze diese kompakte Formel:

1. **Für Vorfilterung, Textnormalisierung und MPN-Abgleich auf der CPU:** Kombination aus `Regex` + `spaCy / Hazm` + `RapidFuzz` + `scikit-learn / BERT-Encoder` (Kapitel 12).
2. **Für Preis-, Bedarfs- und Lieferzeitprognosen auf tabellarischen Daten:** `LightGBM` + `scikit-learn` auf Daten aus `ClickHouse` / `MariaDB` (Kapitel 8 und 28).
3. **Für privates LLM-Serving auf dem GPU-Server:** `vLLM` (oder `SGLang`) für hohen Durchsatz in der Produktion und `llama.cpp` für schlanke Umgebungen, Edge oder Air-gapped-Installationen (Kapitel 21).
4. **Für Dokumenten- und Datenblattsuche (Hybrid RAG):** PDF-Extraktion mit `Docling` + Embedding und Reranking mit `BGE-M3` + Vektordatenbank `Qdrant` oder `Elasticsearch` mit obligatorischem `tenant_id`-Filter (Kapitel 24 und 25).
5. **Für den Bau von Agenten, das Nachladen von `SKILL.md` und parallele Subagenten:** In **PHP / Laravel** mit `Http::pool()` und `Prism`; in **Node.js / TypeScript** mit dem `Vercel AI SDK` und `@modelcontextprotocol/sdk`; und in **Python** mit `PydanticAI` oder `LangGraph` — abgesichert durch `OutboundQuerySanitizer` und die 4-stufige Gedächtnisarchitektur bei externen Recherchen (Kapitel 33 und 34).
6. **Für schnelles, speichersparendes Fine-Tuning eines Verhaltens oder Formats:** Datenaufbereitung und Audit mit den Skripten aus Kapitel 36 + LoRA-/QLoRA-Training mit **`Unsloth`** + Bereitstellung des Adapters über `vLLM` (Kapitel 37).
7. **Für das Release-Gate und die Überwachung im Produktivbetrieb:** Automatisierte Tests und Red-Teaming mit `Promptfoo` in CI/CD + internes Tracing mit `Langfuse` (Kapitel 38 bis 44).
