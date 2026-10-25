# جعبه ابزار عملیاتی و مخازن متن‌باز مهندسی هوش مصنوعی + مخزن کدهای کتاب

!!! goal "در این پیوست"
    - نشانی و ساختار **مخزن عمومی گیت‌هاب این کتاب (`github.com/masmaleki/private-ai-book`)** را می‌بینی تا همه‌ی کدهای Python، Node.js و PHP کتاب را یک‌جا کلون و اجرا کنی.
    - یک **فهرست مرجع و طبقه‌بندی‌شده از بهترین ابزارها و مخازن متن‌باز (Open-Source Repositories) دنیای هوش مصنوعی** در سال ۲۰۲۶ را در اختیار داری که از لایه‌ی سروینگ روی GPU و Fine-tuning تا لایه‌ی RAG، پروتکل MCP، ارکستریشن Agentic AI، ارزیابی و پردازش زبان فارسی را پوشش می‌دهد.
    - برای هر ابزار می‌بینی **کجا به کار می‌آید**، **در کدام فصل کتاب مفاهیمش را یاد گرفته‌ای**، و **در کدام زبان برنامه‌نویسی (Python، TypeScript/Node.js، PHP/Laravel، C++/Go)** قابل استفاده است.

## ۱. مخزن رسمی کدها و فهرست زنده‌ی ابزارهای کتاب در گیت‌هاب

تمام اسکریپت‌های اجرایی این کتاب، تست‌های خودکار (`pytest`)، پیاده‌سازی‌های چندزبانه در **Python**، **Node.js (TypeScript/ESM)** و **PHP 8.3 (Laravel)**، و نسخه‌ی زنده‌ و به‌روزشونده‌ی جداول همین پیوست، در مخزن عمومی زیر در دسترس شماست:

- **آدرس مخزن گیت‌هاب کتاب:**  
  [`https://github.com/masmaleki/private-ai-book`](https://github.com/masmaleki/private-ai-book)

برای دریافت و اجرای سریع کدها روی سیستم خودت:

```bash
git clone https://github.com/masmaleki/private-ai-book.git
cd private-ai-book

# اجرای تست‌های پایپ‌لاین ترکیبی NLP (فصل ۱۲) و معماری Agentic AI (فصل ۳۴)
pytest code/ch12/ code/ch34/ -v

# اجرای ارکستریتور ریزعامل‌ها در Node.js
node code/ch34/agentic_node.mjs
```

چون دنیای ابزارهای متن‌باز هوش مصنوعی هر ماه در حال تکامل است، هر زمان کتابخانه‌ی مهم جدیدی منتشر شود یا نسخه‌ی جدیدی از ابزارهای معرفی‌شده در این پیوست بیاید، جدول `README.md` در مخزن بالا به‌روزرسانی می‌شود.

---

## ۲. لایه‌ی اول: سروینگ و اجرای خصوصی مدل‌های زبانی (Inference & Model Serving)

وقتی می‌خواهی یک مدل Open-weight (مثل خانواده‌ی `Qwen`، `Llama`، `Mistral` یا `Gemma`) را روی سرورهای داخلی خودت (فصل ۲۱ و ۴۱) بالا بیاوری و یک API استاندارد سازگار با OpenAI تحویل لایه‌ی Backend بدهی، این پنج مخزن ستون فقرات صنعتی تو هستند:

| نام ابزار / مخزن | آدرس گیت‌هاب | کاربرد اصلی و نقطه‌ی قوت | فصل مرجع |
|---|---|---|---|
| **vLLM** | [`github.com/vllm-project/vllm`](https://github.com/vllm-project/vllm) | استاندارد طلایی سروینگ صنعتی LLM روی GPU؛ مجهز به **PagedAttention**، **Continuous Batching**، سروینگ هم‌زمان چند **LoRA Adapter** و خروجی ساختاریافته. | ۱۸، ۲۱، ۳۷ |
| **llama.cpp** | [`github.com/ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) | موتور فوق‌سریع C/C++ بدون وابستگی سنگین، ایده‌آل برای فرمت **GGUF**، اجرای ترکیبی CPU+GPU، ایستگاه‌های کاری، سرورهای سبک و محیط‌های Air-gapped. | ۱۹، ۲۱، ۴۱ |
| **SGLang** | [`github.com/sgl-project/sglang`](https://github.com/sgl-project/sglang) | موتور سروینگ فوق‌سریع با **RadixAttention** (کش خودکار پیشوندهای مشترک پرامپت) و دیکودینگ قیددار (JSON Schema با سربار نزدیک به صفر)؛ عالی برای سیستم‌های چندعاملی که پیشوند پرامپت تکراری دارند. | ۱۸، ۲۱، ۲۲ |
| **Text Generation Inference (TGI)** | [`github.com/huggingface/text-generation-inference`](https://github.com/huggingface/text-generation-inference) | سرور رسمی شرکت Hugging Face با هسته‌ی Rust/Python برای سروینگ پایدار مدل‌ها در محیط‌های کانتینری. | ۲۱ |
| **Ollama** | [`github.com/ollama/ollama`](https://github.com/ollama/ollama) | ساده‌ترین ابزار (مبتنی بر Go و `llama.cpp`) برای بالا آوردن محلی مدل‌ها با یک دستور روی لپ‌تاپ توسعه‌دهندگان در فاز نمونه‌سازی. | ۲۰، ۲۱ |

جدول هـ-۱: مخازن مرجع لایه‌ی سروینگ و استنتاج خصوصی.

---

## ۳. لایه‌ی دوم: آموزش، Fine-tuning، LoRA/QLoRA و هم‌راستاسازی (Training & Fine-Tuning)

وقتی طبق درخت تصمیم فصل ۳۵ به این نتیجه رسیدی که مدل باید یک «رفتار، قالب یا لحن» جدید یاد بگیرد، این ابزارها (فصل ۳۶ و ۳۷) هزینه‌ی GPU و زمان آموزش را به شکل چشمگیری کم می‌کنند:

| نام ابزار / مخزن | آدرس گیت‌هاب | کاربرد اصلی و نقطه‌ی قوت | فصل مرجع |
|---|---|---|---|
| **Unsloth** | [`github.com/unslothai/unsloth`](https://github.com/unslothai/unsloth) | کرنل‌های سفارشی **Triton** و مشتق‌گیری دستی گرادیان؛ **۲ تا ۵ برابر آموزش سریع‌تر** و **تا ۷۰٪ مصرف VRAM کمتر** در LoRA، QLoRA و GRPO با صفر درصد افت دقت و خروجی مستقیم برای `vLLM` و `GGUF`. | ۳۷ |
| **Hugging Face PEFT** | [`github.com/huggingface/peft`](https://github.com/huggingface/peft) | کتابخانه‌ی مرجع روش‌های کم‌پارامتر (LoRA، DoRA، QLoRA، IA3) که با تمام اکوسیستم Transformers یکپارچه است. | ۳۷ |
| **Hugging Face TRL** | [`github.com/huggingface/trl`](https://github.com/huggingface/trl) | آموزش‌دهنده‌های استاندارد پس از پیش‌آموزش شامل `SFTTrainer`، هم‌راستاسازی `DPOTrainer` و آموزش استدلالی `GRPOTrainer`. | ۱۷، ۳۷ |
| **Axolotl** | [`github.com/axolotl-ai-cloud/axolotl`](https://github.com/axolotl-ai-cloud/axolotl) | چارچوب Fine-tuning مبتنی بر فایل پیکربندی YAML؛ عالی برای تیم‌هایی که می‌خواهند روی چند GPU (با DeepSpeed و FSDP) بدون نوشتن کد تکراری آموزش دهند. | ۳۷ |
| **LLaMA-Factory** | [`github.com/hiyouga/LLaMA-Factory`](https://github.com/hiyouga/LLaMA-Factory) | چارچوب جامع Fine-tuning برای بیش از ۱۰۰ مدل باز، همراه با رابط گرافیکی WebUI (برای آزمایش سریع) و CLI صنعتی. | ۳۷ |
| **bitsandbytes** | [`github.com/bitsandbytes-foundation/bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) | کتابخانه‌ی پایه‌ی کوانتیزاسیون ۴ بیتی (`NF4`) و ۸ بیتی و بهینه‌سازهای Paged برای اجرای QLoRA روی کارت‌های گرافیک مصرفی. | ۱۹، ۳۷ |

جدول هـ-۲: مخازن مرجع لایه‌ی Fine-tuning و فشرده‌سازی آموزش.

---

## ۴. لایه‌ی سوم: هوش مصنوعی عاملی، MCP و ارکستریشن چندزبانه (Agentic AI & MCP)

همان‌طور که در فصل ۲۳، ۳۳ و ۳۴ دیدیم، لایه‌ی عاملی (مغز + دست‌ها + سیستم عصبی MCP + تیم ریزعامل‌ها) به هیچ وجه محدود به پایتون نیست و می‌توانی آن را در زبان اصلی Backend خودت بسازی:

| نام ابزار / مخزن | آدرس گیت‌هاب | زبان | کاربرد اصلی و نقطه‌ی قوت | فصل مرجع |
|---|---|---|---|---|
| **Model Context Protocol (MCP)** | [`github.com/modelcontextprotocol`](https://github.com/modelcontextprotocol) | Python, TS, C#, Go, Java | مخزن رسمی استاندارد **MCP** (سیستم عصبی استک هوش مصنوعی) شامل SDKهای رسمی برای ساخت سرور و کلاینت ابزارها. | ۲۳، ۳۴ |
| **PydanticAI** | [`github.com/pydantic/pydantic-ai`](https://github.com/pydantic/pydantic-ai) | Python | چارچوب ساخت Agent با اعتبارسنجی سخت‌گیرانه‌ی نوع داده (Type-safe)، تزریق وابستگی و خروجی ساختاریافته توسط تیم سازنده‌ی Pydantic. | ۲۲، ۳۳، ۳۴ |
| **LangGraph** | [`github.com/langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph) | Python, TS | ساخت سیستم‌های عاملی و چندعاملی به صورت **گراف وضعیت‌دار (Stateful Graph)** با پشتیبانی عالی از دروازه‌ی تأیید انسانی (Human-in-the-loop) و بازنگری (Reflection). | ۳۳، ۳۴ |
| **OpenAI Agents SDK** | [`github.com/openai/openai-agents-python`](https://github.com/openai/openai-agents-python) | Python | چارچوب سبک و مینیمال برای تعریف Agentها، واگذاری کار به ریزعامل‌ها (**Handoffs / Subagents**)، Guardrailها و ردیابی (Tracing) که با هر سرور سازگار با OpenAI مثل `vLLM` هم کار می‌کند. | ۳۳، ۳۴ |
| **Vercel AI SDK** | [`github.com/vercel/ai`](https://github.com/vercel/ai) | TypeScript / Node.js | محبوب‌ترین جعبه‌ابزار مهندسی هوش مصنوعی در دنیای **Node.js و TypeScript**؛ پشتیبانی یکپارچه از خروجی ساختاریافته‌ی Zod، فراخوانی ابزار، حلقه‌های Agentic و اتصال به مدل‌های خصوصی. | ۲۲، ۳۳، ۳۴ |
| **Prism (Laravel AI)** | [`github.com/echolabsdev/prism`](https://github.com/echolabsdev/prism) | PHP / Laravel | پکیج بومی و مدرن **Laravel** برای کار با LLMها، تعریف Tools، خروجی ساختاریافته (Structured Output) و اتصال به سرورهای خصوصی یا ابری به سبک اصیل لاراول. | ۲۲، ۳۳، ۳۴ |
| **OpenAI PHP Client** | [`github.com/openai-php/client`](https://github.com/openai-php/client) | PHP / Laravel | کلاینت استاندارد و بدون وابستگی PHP که با تغییر `baseUri` مستقیماً به سرور داخلی `vLLM` یا `Ollama` وصل می‌شود. | ۲۱، ۳۴ |

جدول هـ-۳: مخازن مرجع لایه‌ی Agentic AI و پروتکل MCP در Python، Node.js و PHP.

---

## ۵. لایه‌ی چهارم: RAG، استخراج اسناد، پایگاه‌های داده‌ی برداری و تحلیلی

در فصل ۲۴، ۲۵ و ۲۸ دیدیم که کیفیت پاسخ سیستم مستقیماً به کیفیت «استخراج متن از PDF/جداول» و «جست‌وجوی ترکیبی + تحلیل عددی» وابسته است:

| نام ابزار / مخزن | آدرس گیت‌هاب | کاربرد اصلی و نقطه‌ی قوت | فصل مرجع |
|---|---|---|---|
| **Docling** | [`github.com/docling-project/docling`](https://github.com/docling-project/docling) | ابزار متن‌باز فوق‌العاده (توسعه‌یافته در IBM Research) برای تبدیل PDFهای پیچیده، دیتاشیت‌ها و جداول مهندسی به Markdown و JSON تمیز، آماده برای تقطیع (Chunking) در RAG. | ۲۴، ۲۵ |
| **Qdrant** | [`github.com/qdrant/qdrant`](https://github.com/qdrant/qdrant) | موتور جست‌وجوی برداری و ترکیبی (Sparse + Dense) نوشته‌شده با Rust؛ بسیار سریع، کم‌مصرف و مجهز به فیلترینگ دقیق Payload برای جداسازی سخت‌گیرانه‌ی Tenantها (`tenant_id`). | ۱۴، ۲۴، ۲۵ |
| **ClickHouse** | [`github.com/ClickHouse/ClickHouse`](https://github.com/ClickHouse/ClickHouse) | پایگاه داده‌ی ستونی فوق‌سریع برای کوئری‌های تحلیلی زنجیره‌ی تأمین، محاسبه‌ی صدک‌های قیمت (`P10/P50/P90`)، لاگ‌های ممیزی و ذخیره‌ی تله‌متری LLM. | ۱، ۲۳، ۲۸، ۴۴ |
| **chDB** | [`github.com/chdb-io/chdb`](https://github.com/chdb-io/chdb) | موتور SQL درون‌پردازشی (In-process) ClickHouse برای پایتون و Node.js؛ اجرای مستقیم کوئری‌های تحلیلی سنگین روی فایل‌های Parquet، CSV و JSON بدون نیاز به سرور جداگانه. | ۶، ۲۸ |
| **Sentence-Transformers** | [`github.com/UKPLab/sentence-transformers`](https://github.com/UKPLab/sentence-transformers) | کتابخانه‌ی استاندارد اجرای محلی مدل‌های **Embedding** (مثل `bge-m3` و `multilingual-e5`) و مدل‌های **Cross-Encoder Reranker** برای جست‌وجوی چندزبانه. | ۱۲، ۱۴، ۲۴، ۲۵ |
| **FlagEmbedding (BGE)** | [`github.com/FlagOpen/FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding) | مخزن مدل‌های `BGE-M3` (پشتیبانی هم‌زمان از Dense، Sparse/Lexical و Multi-Vector در بیش از ۱۰۰ زبان از جمله فارسی، آلمانی و انگلیسی) و Rerankerهای دقیق. | ۱۴، ۲۴، ۲۵ |

جدول هـ-۴: مخازن مرجع لایه‌ی RAG، استخراج اسناد و موتورهای داده.

---

## ۶. لایه‌ی پنجم: پردازش زبان طبیعی کلاسیک، فارسی و یادگیری ماشین جدولی

قبل از بیدار کردن LLM (فصل ۱۲) و برای پیش‌بینی‌های عددی و سری زمانی (فصل ۷ تا ۱۰ و ۲۸)، این ابزارها قهرمان‌های بی‌رقیب روی CPU هستند:

| نام ابزار / مخزن | آدرس گیت‌هاب | کاربرد اصلی و نقطه‌ی قوت | فصل مرجع |
|---|---|---|---|
| **spaCy** | [`github.com/explosion/spaCy`](https://github.com/explosion/spaCy) | کتابخانه‌ی صنعتی و فوق‌سریع NLP در پایتون/Cython برای توکن‌سازی، POS Tagging، تجزیه‌ی وابستگی و پایپ‌لاین‌های NER سفارشی روی CPU. | ۱۲ |
| **RapidFuzz** | [`github.com/rapidfuzz/RapidFuzz`](https://github.com/rapidfuzz/RapidFuzz) | پیاده‌سازی C++ فوق‌سریع فاصله‌ی ویرایشی **Levenshtein**، **Jaro-Winkler** و تطبیق فازی رشته‌ها؛ ایده‌آل برای تمیزسازی کد قطعات (MPN) و نام شرکت‌ها در مقیاس میلیونی. | ۱۲، ۲۶ |
| **Hazm (هضم)** | [`github.com/roshan-research/hazm`](https://github.com/roshan-research/hazm) | کتابخانه‌ی شناخته‌شده‌ی پردازش زبان فارسی برای نرمال‌سازی متن فارسی، اصلاح نیم‌فاصله‌ها، ریشه‌یابی (Stemming/Lemmatization) و برچسب‌زنی نقش کلمات. | ۱۲ |
| **ParsBERT** | [`github.com/hooshvare/parsbert`](https://github.com/hooshvare/parsbert) | خانواده‌ی مدل‌های ترنسفورمر **Encoder-Only** پیش‌آموزش‌دیده روی پیکره‌های بزرگ فارسی برای دسته‌بندی متن، تحلیل احساس و شناسایی موجودیت‌های نام‌دار (NER) فارسی. | ۱۲ |
| **Hugging Face Tokenizers** | [`github.com/huggingface/tokenizers`](https://github.com/huggingface/tokenizers) | پیاده‌سازی Rust الگوریتم‌های BPE، WordPiece و Unigram برای ساخت و اجرای توکن‌سازهای فوق‌سریع. | ۱۳ |
| **scikit-learn** | [`github.com/scikit-learn/scikit-learn`](https://github.com/scikit-learn/scikit-learn) | جعبه‌ابزار استاندارد یادگیری ماشین کلاسیک (رگرسیون، درخت تصمیم، Random Forest، TF-IDF، Naive Bayes، خوشه‌بندی و معیارهای ارزیابی). | ۶ تا ۱۰، ۱۲ |
| **LightGBM** | [`github.com/microsoft/LightGBM`](https://github.com/microsoft/LightGBM) | موتور Gradient Boosted Decision Trees فوق‌سریع؛ انتخاب اول صنعتی برای پیش‌بینی قیمت، رگرسیون چندصدکی (`P10/P50/P90`) و پیش‌بینی Lead Time روی داده‌های جدولی ERP. | ۸، ۲۸، ۲۹ |

جدول هـ-۵: مخازن مرجع لایه‌ی NLP کلاسیک/فارسی و یادگیری ماشین جدولی.

---

## ۷. لایه‌ی ششم: ارزیابی، امنیت، Red-Teaming و مشاهده‌پذیری (Eval, Security & Observability)

هیچ سیستم هوش مصنوعی بدون دروازه‌ی ارزیابی خودکار (فصل ۳۸ و ۳۹)، تست نفوذ (فصل ۴۰) و ردیابی در تولید (فصل ۴۴) اجازه‌ی ورود به محیط عملیاتی را ندارد:

| نام ابزار / مخزن | آدرس گیت‌هاب | کاربرد اصلی و نقطه‌ی قوت | فصل مرجع |
|---|---|---|---|
| **Promptfoo** | [`github.com/promptfoo/promptfoo`](https://github.com/promptfoo/promptfoo) | ابزار خط فرمان (CLI) فوق‌العاده برای تست رگرسیون پرامپت‌ها، مقایسه‌ی مدل‌ها در CI/CD و اجرای خودکار حملات **Red-Teaming** (تزریق پرامپت، نشت داده و Jailbreak). | ۳۸، ۳۹، ۴۰ |
| **Langfuse** | [`github.com/langfuse/langfuse`](https://github.com/langfuse/langfuse) | پلتفرم متن‌باز و قابل میزبانی داخلی (**Self-hosted**) برای مشاهده‌پذیری (Observability)، ثبت Trace گام‌به‌گام ریزعامل‌ها، مدیریت نسخه‌ی پرامپت‌ها و رصد هزینه/تأخیر. | ۳۴، ۳۸، ۴۴ |
| **Arize Phoenix** | [`github.com/Arize-ai/phoenix`](https://github.com/Arize-ai/phoenix) | پلتفرم متن‌باز ردیابی (مبتنی بر OpenTelemetry) و ارزیابی سیستم‌های LLM، RAG و Agentها. | ۳۸، ۴۴ |
| **Ragas** | [`github.com/explodinggradients/ragas`](https://github.com/explodinggradients/ragas) | چارچوب تخصصی ارزیابی پایپ‌لاین‌های **RAG** (سنجش وفاداری پاسخ به منبع `Faithfulness`، دقت بازیابی `Context Precision` و پوشش پاسخ). | ۲۵، ۳۸ |
| **Microsoft Presidio** | [`github.com/microsoft/presidio`](https://github.com/microsoft/presidio) | موتور متن‌باز شناسایی و ناشناس‌سازی اطلاعات شخصی (**PII Redaction & Pseudonymization**) در متن‌ها و لاگ‌ها پیش از ذخیره‌سازی یا Fine-tuning. | ۳۶، ۴۰، ۴۲ |
| **LLM Guard** | [`github.com/protectai/llm-guard`](https://github.com/protectai/llm-guard) | مجموعه‌ای از اسکنرهای امنیتی سریع برای ورودی و خروجی LLM (تشخیص Prompt Injection، نشت اسرار، سانسور PII و اعتبارسنجی خروجی). | ۴۰ |

جدول هـ-۶: مخازن مرجع لایه‌ی ارزیابی، امنیت و مشاهده‌پذیری.

---

## ۸. نقشه‌ی یک‌نگاهی انتخاب ابزار برای پروژه‌ی بعدی تو

هر زمان در جلسه‌ی معماری خواستی تصمیم بگیری کدام مخزن را برای کدام لایه انتخاب کنی، از این فرمول خلاصه استفاده کن:

1. **برای فیلتر اولیه، نرمال‌سازی فارسی و تطبیق کد قطعه روی CPU:** ترکیب `Regex` + `Hazm` + `RapidFuzz` + `scikit-learn / ParsBERT` (فصل ۱۲).
2. **برای پیش‌بینی قیمت، تقاضا و زمان تحویل روی داده‌ی جدولی:** `LightGBM` + `scikit-learn` روی داده‌های `ClickHouse` / `MariaDB` (فصل ۸ و ۲۸).
3. **برای سروینگ خصوصی LLM روی سرور GPU:** `vLLM` (یا `SGLang`) برای تولید با توان عملیاتی بالا، و `llama.cpp` برای محیط‌های سبک یا CPU/Edge (فصل ۲۱).
4. **برای جست‌وجوی اسناد و دیتاشیت‌ها (RAG):** استخراج PDF با `Docling` + مدل امبدینگ و بازرتبه‌بندی `BGE-M3` + پایگاه داده‌ی `Qdrant` یا `Elasticsearch` با فیلتر اجباری `tenant_id` (فصل ۲۴ و ۲۵).
5. **برای ساخت Agent، بارگذاری `SKILL.md` و ریزعامل‌های موازی:** در **PHP/Laravel** با `Http::pool()` و `Prism`؛ در **Node.js/TypeScript** با `Vercel AI SDK` و `@modelcontextprotocol/sdk`؛ و در **Python** با `PydanticAI` یا `LangGraph` (فصل ۳۳ و ۳۴).
6. **برای Fine-tuning سریع و کم‌مصرف یک رفتار یا قالب:** آماده‌سازی و ممیزی داده با اسکریپت‌های فصل ۳۶ + آموزش LoRA/QLoRA با **`Unsloth`** + سرو کردن Adapter روی `vLLM` (فصل ۳۷).
7. **برای دروازه‌ی انتشار و پایش در تولید:** تست خودکار و Red-teaming با `Promptfoo` در CI/CD + ردیابی داخلی با `Langfuse` (فصل ۳۸ تا ۴۴).
