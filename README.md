# هوش خصوصی: از توکن تا تصمیم (Private AI: From Token to Decision — ویرایش پنجم / Revision 5)

**مخزن رسمی کدهای همراه کتاب و فهرست زنده‌ی ابزارها و مخازن متن‌باز مهندسی هوش مصنوعی**  
**نویسنده:** محمدصادق مالکی (Mohammad Sadegh Maleki) — هم‌بنیان‌گذار و CTO شرکت AustroByte GmbH  
**آدرس مخزن:** [`https://github.com/masmaleki/private-ai-book`](https://github.com/masmaleki/private-ai-book)

---

## معرفی کوتاه (Overview)

این مخزن شامل دو بخش اصلی است:

1. **کدهای اجرایی تمام فصل‌های کتاب (`code/`):**
   - اسکریپت‌های **Python 3.11+** برای یادگیری ماشین از صفر، پردازش زبان طبیعی (NLP)، درون LLM، سروینگ خصوصی، RAG، MCP، پیش‌بینی قیمت/تقاضا، Agentic AI، Fine-tuning با LoRA/QLoRA و امنیت.
   - پیاده‌سازی‌های چندزبانه‌ی **Agentic AI (فصل ۳۴)** در **Python** (`code/ch34/agentic_system.py`)، **Node.js / TypeScript** (`code/ch34/agentic_node.mjs`) و **PHP 8.3 / Laravel** (`code/ch34/AgenticQuoteOrchestrator.php`).
   - پروژه‌ی مرجع قابل اجرای پیوست ج (`code/appC/`).
2. **فهرست طبقه‌بندی‌شده‌ی بهترین مخازن متن‌باز مهندسی هوش مصنوعی (پیوست هـ کتاب):**
   - از موتورهای سروینگ (`vLLM`، `llama.cpp`، `SGLang`) و شتاب‌دهنده‌ی آموزش (`Unsloth`) تا چارچوب‌های عاملی (`MCP`، `PydanticAI`، `LangGraph`، `Vercel AI SDK`، `Laravel Prism`)، پایگاه‌های داده (`Qdrant`، `ClickHouse`) و ابزارهای پردازش زبان فارسی (`Hazm`، `ParsBERT`).

---

## استک هوش مصنوعی به زبان بدن انسان (Your AI Stack, Explained Like a Human Body — فصل ۳۴)

| لایه در استک | معادل در بدن انسان | نقش اصلی |
|---|---|---|
| **1. LLM** | **مغز (`The Brain`)** | می‌فهمد، می‌نویسد و استدلال می‌کند؛ فقط چیزهایی را می‌داند که در زمان آموزش یاد گرفته است. |
| **2. RAG** | **مغز + کتابخانه (`Brain + Library`)** | قبل از پاسخ دادن اسناد و مدارک شما را جست‌وجو می‌کند و پاسخ را به داده‌ی واقعی مستند می‌سازد. |
| **3. MCP** | **سیستم عصبی (`The Nervous System`)** | یک درگاه استاندارد واحد برای اتصال امن هوش مصنوعی به نرم‌افزارها، APIها، فایل‌ها و دیتابیس‌ها. |
| **4. Skills (`SKILL.md`)** | **حافظه‌ی عضلانی (`Muscle Memory`)** | دانش و روش کار ذخیره‌شده (SOP) برای کارهای تکرارشونده؛ فقط زمانی که کار به آن نیاز دارد بارگذاری می‌شود (Progressive Disclosure). |
| **5. AI Agent** | **مغز + دست‌ها (`Brain + Hands`)** | مراحل کار را برنامه‌ریزی می‌کند و از ابزارها برای اقدام و عمل استفاده می‌کند، نه فقط حرف زدن. |
| **6. Subagents** | **تیم متخصصان با میز کار مجزا (`Team of Specialists`)** | هر متخصص در پنجره‌ی زمینه‌ی ایزوله‌ی خودش (Context Isolation) کار را موازی پیش می‌برد و فقط خلاصه‌ی تمیز را برمی‌گرداند. |

---

## اجرای سریع کدها (Quickstart)

```bash
# 1. کلون کردن مخزن
git clone https://github.com/masmaleki/private-ai-book.git
cd private-ai-book

# 2. اجرای پایپ‌لاین ترکیبی NLP و تست‌های خودکار (فصل ۱۲)
python3 code/ch12/nlp_pipeline.py
pytest code/ch12/test_nlp_pipeline.py -v

# 3. اجرای سیستم Agentic AI (Skills + Subagents + Web Research + 4-Tier Memory + Reflection) در پایتون، Node.js و تست‌ها (فصل ۳۴)
python3 code/ch34/agentic_system.py
python3 code/ch34/web_research_memory_agent.py
pytest code/ch34/test_agentic_system.py -v
node code/ch34/agentic_node.mjs
```

---

## فهرست مرجع مخازن متن‌باز مهندسی هوش مصنوعی (Curated Open-Source AI Toolkit)

### ۱. سروینگ و اجرای خصوصی مدل‌ها (Inference & Serving)
- **[vLLM](https://github.com/vllm-project/vllm)** — استاندارد صنعتی سروینگ LLM روی GPU با PagedAttention و Multi-LoRA (فصل ۱۸، ۲۱، ۳۷).
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — موتور C/C++ فوق‌سریع برای فرمت GGUF روی CPU/GPU و محیط‌های Air-gapped (فصل ۱۹، ۲۱، ۴۱).
- **[SGLang](https://github.com/sgl-project/sglang)** — سروینگ سریع با RadixAttention (کش خودکار پیشوند) و خروجی ساختاریافته (فصل ۱۸، ۲۱، ۲۲).
- **[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)** — سرور کانتینری Hugging Face (فصل ۲۱).
- **[Ollama](https://github.com/ollama/ollama)** — اجرای محلی سریع مدل‌ها برای توسعه‌دهندگان (فصل ۲۰، ۲۱).

### ۲. آموزش و Fine-tuning سریع (LoRA / QLoRA / RL)
- **[Unsloth](https://github.com/unslothai/unsloth)** — کرنل‌های سفارشی Triton و مشتق‌گیری دستی گرادیان؛ **۲ تا ۵ برابر آموزش سریع‌تر و تا ۷۰٪ VRAM کمتر** برای LoRA، QLoRA و GRPO با صفر درصد افت دقت (فصل ۳۷).
- **[Hugging Face PEFT](https://github.com/huggingface/peft)** — کتابخانه‌ی مرجع LoRA و QLoRA (فصل ۳۷).
- **[Hugging Face TRL](https://github.com/huggingface/trl)** — آموزش SFT، DPO و GRPO (فصل ۱۷، ۳۷).
- **[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)** — چارچوب Fine-tuning مبتنی بر YAML برای تک‌GPU و کلاسترهای چندگرهی (فصل ۳۷).
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — پلتفرم جامع Fine-tuning با CLI و WebUI (فصل ۳۷).
- **[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)** — کوانتیزاسیون ۴ بیتی NF4 و بهینه‌سازهای Paged (فصل ۱۹، ۳۷).

### ۳. هوش مصنوعی عاملی، MCP و ارکستریشن در Python، Node.js و PHP
- **[Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)** — استاندارد و SDKهای رسمی اتصال ابزارها در Python، TypeScript، C# و Go (فصل ۲۳، ۳۴).
- **[PydanticAI](https://github.com/pydantic/pydantic-ai)** — چارچوب Agentic نوع‌امن در پایتون (فصل ۲۲، ۳۳، ۳۴).
- **[LangGraph](https://github.com/langchain-ai/langgraph)** — ارکستریشن گراف وضعیت‌دار برای سیستم‌های چندعاملی و Human-in-the-loop (فصل ۳۳، ۳۴).
- **[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)** — چارچوب سبک برای Agentها و Handoffs/Subagents سازگار با vLLM (فصل ۳۳، ۳۴).
- **[Vercel AI SDK](https://github.com/vercel/ai)** — جعبه‌ابزار استاندارد **TypeScript / Node.js** برای خروجی ساختاریافته، ابزارها و حلقه‌های عاملی (فصل ۲۲، ۳۳، ۳۴).
- **[Prism (Laravel AI)](https://github.com/echolabsdev/prism)** — پکیج بومی **PHP / Laravel** برای LLM، Tools و Structured Output (فصل ۲۲، ۳۳، ۳۴).
- **[OpenAI PHP Client](https://github.com/openai-php/client)** — کلاینت استاندارد PHP سازگار با سرورهای داخلی vLLM (فصل ۲۱، ۳۴).

### ۴. RAG، استخراج اسناد و پایگاه‌های داده
- **[Docling](https://github.com/docling-project/docling)** — استخراج دقیق PDFهای پیچیده و جداول دیتاشیت به Markdown/JSON (فصل ۲۴، ۲۵).
- **[Qdrant](https://github.com/qdrant/qdrant)** — موتور جست‌وجوی برداری و ترکیبی با فیلترینگ سریع `tenant_id` (فصل ۱۴، ۲۴، ۲۵).
- **[ClickHouse](https://github.com/ClickHouse/ClickHouse)** — پایگاه داده‌ی تحلیلی ستونی فوق‌سریع برای قیمت‌گذاری، صدک‌ها و تله‌متری (فصل ۱، ۲۸، ۴۴).
- **[chDB](https://github.com/chdb-io/chdb)** — موتور SQL درون‌پردازشی ClickHouse روی فایل‌های Parquet/CSV/JSON (فصل ۶، ۲۸).
- **[Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)** و **[FlagEmbedding (BGE-M3)](https://github.com/FlagOpen/FlagEmbedding)** — مدل‌های Embedding و Reranker چندزبانه (فصل ۱۲، ۱۴، ۲۴، ۲۵).

### ۵. پردازش زبان طبیعی (NLP) کلاسیک/فارسی و یادگیری ماشین جدولی
- **[spaCy](https://github.com/explosion/spaCy)** — پایپ‌لاین صنعتی و سریع NLP روی CPU (فصل ۱۲).
- **[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)** — تطبیق فازی فوق‌سریع رشته‌ها (Levenshtein / Jaro-Winkler) برای کد قطعات MPN (فصل ۱۲، ۲۶).
- **[Hazm (هضم)](https://github.com/roshan-research/hazm)** — نرمال‌سازی متن فارسی، نیم‌فاصله و ریشه‌یابی (فصل ۱۲).
- **[ParsBERT](https://github.com/hooshvare/parsbert)** — مدل‌های ترنسفورمر Encoder-Only برای دسته‌بندی و NER فارسی (فصل ۱۲).
- **[Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)** — توکن‌سازهای سریع Rust (فصل ۱۳).
- **[scikit-learn](https://github.com/scikit-learn/scikit-learn)** و **[LightGBM](https://github.com/microsoft/LightGBM)** — یادگیری ماشین کلاسیک و Gradient Boosting جدولی (فصل ۶ تا ۱۰، ۲۸، ۲۹).

### ۶. ارزیابی، امنیت و مشاهده‌پذیری (Eval, Security & Observability)
- **[Promptfoo](https://github.com/promptfoo/promptfoo)** — ارزیابی خودکار در CI/CD و تست نفوذ Red-Teaming (فصل ۳۸، ۳۹، ۴۰).
- **[Langfuse](https://github.com/langfuse/langfuse)** — مشاهده‌پذیری و ردیابی (Tracing) خودمیزبان برای سیستم‌های LLM و ریزعامل‌ها (فصل ۳۴، ۳۸، ۴۴).
- **[Arize Phoenix](https://github.com/Arize-ai/phoenix)** — ردیابی OpenTelemetry و ارزیابی RAG/Agents (فصل ۳۸، ۴۴).
- **[Ragas](https://github.com/explodinggradients/ragas)** — ارزیابی تخصصی پایپ‌لاین‌های RAG (فصل ۲۵، ۳۸).
- **[Microsoft Presidio](https://github.com/microsoft/presidio)** و **[LLM Guard](https://github.com/protectai/llm-guard)** — شناسایی PII و دفاع در برابر تزریق پرامپت (فصل ۳۶، ۴۰، ۴۲).

---

## ساختار کامل مخزن (Repository Structure)

- **[`CHAPTER_INDEX.md`](./CHAPTER_INDEX.md)** — جدول کامل تناظر تمام **۴۵ فصل و ۵ پیوست** کتاب با پوشه‌های کد و اسکریپت‌های اجرایی.
- **[`TOOLKIT_FA.md`](./TOOLKIT_FA.md)** — راهنمای جامع و طبقه‌بندی‌شده‌ی ابزارها و مخازن متن‌باز مهندسی هوش مصنوعی (پیوست هـ کتاب).
- **[`code/`](./code/)** — کدهای اجرایی فصل‌به‌فصل (`ch01` تا `ch45`) و پیوست‌ها (`appA`، `appC`، `appD`):
  - `code/ch12/` — پایپ‌لاین ترکیبی ۳ لایه‌ی پردازش زبان طبیعی فارسی و انگلیسی (`nlp_pipeline.py` + `test_nlp_pipeline.py`).
  - `code/ch34/` — معماری کامل **Agentic AI** (مهارت‌ها `Skills`، ریزعامل‌ها `Subagents`، **ریزعامل جست‌وجوی وب و تحقیق خودکار با سانسور کوئری خروجی** `OutboundQuerySanitizer`، **معماری چهارلایهٔ حافظه و ماندگاری دانش** `MultiTierAgentMemory` و حلقه‌ی `Plan-and-Execute + Reflection`) در سه زبان:
    - **Python**: `code/ch34/agentic_system.py`، `code/ch34/web_research_memory_agent.py` و `code/ch34/test_agentic_system.py`
    - **Node.js / ESM**: `code/ch34/agentic_node.mjs`
    - **PHP 8.3 / Laravel**: `code/ch34/AgenticQuoteOrchestrator.php`
    - **نمونه پروندهٔ تحقیق اپیزودیک (Tier 2 Dossier)**: `code/ch34/research_dossiers/STM32N657X0H6Q_research_dossier.md`
  - `code/ch37/` — آموزش LoRA از صفر (`lora_by_hand.py`، `train_lora.py`) + اسکریپت شتاب‌دهی QLoRA با **Unsloth** (`unsloth_qlora_train.py`).
  - `code/appC/` — پروژه‌ی مرجع یکپارچه‌ی استخراج، قیمت‌گذاری و ارزیابی (پیوست ج).
- **[`skills/`](./skills/)** — کتابخانه‌ی مهارت‌های استاندارد (`SKILL.md`) با الگوی افشای تدریجی (Progressive Disclosure — فصل ۳۴):
  - `skills/rfq-pricing-audit/SKILL.md` — ممیزی قیمت‌گذاری و کف حاشیه‌ی سود بر اساس سطح مشتری.
  - `skills/bom-mpn-normalization/SKILL.md` — نرمال‌سازی کد قطعات (MPN) و پسوندهای بسته‌بندی در لیست قطعات (BOM).
  - `skills/persian-hybrid-nlp-triage/SKILL.md` — تریاژ ۳ لایه‌ی پیام‌های بازرگانی فارسی و دوزبانه.
  - `skills/security-egress-and-pii-guard/SKILL.md` — کنترل خروج داده (Zero Unauthorized Egress)، جداسازی سازمان‌ها و حذف PII.
  - `skills/web-deep-research-guard/SKILL.md` — پروتکل تحقیق وب بدون نشت اسرار تجاری، رتبه‌بندی اعتبار دامنه و ثبت در حافظهٔ چهارلایه.
- **[`assets/diagrams/`](./assets/diagrams/)** — دیاگرام‌های معماری با کیفیت بالا (از جمله اینفوگرافیک *«استک هوش مصنوعی به زبان بدن انسان»* در `assets/diagrams/ch34/ai_stack_human_body.png` و دیاگرام *«معماری ریزعامل تحقیق وب و چهار لایهٔ ماندگاری حافظه»* در `assets/diagrams/ch34/web_research_memory_tiers.png`).
- **[`chapters/ch34.md`](./chapters/ch34.md)** — متن کامل فصل ۳۴ (ویرایش پنجم) شامل بخش‌های جدید ریزعامل جست‌وجوی وب و معماری چهارلایهٔ حافظه.
- **فایل‌های پیکربندی چندزبانه:** `requirements.txt` (پایتون)، `package.json` (Node.js) و `composer.json` (PHP/Laravel).
