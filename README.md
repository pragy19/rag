# 📚 RAG Demo — Supabase • Nomic • Cohere • Groq LLaMA-3 • Streamlit
[![Live Demo](https://img.shields.io/badge/Demo-Live-green)](https://your-demo-url.com)

A small **Retrieval-Augmented Generation (RAG)** application.  
Users can upload or paste text, which is chunked, embedded, stored in a **Supabase (pgvector)** database, retrieved and reranked, and then answered by **Groq LLaMA-3** with **inline citations**.

---

## ✨ Features

- 📂 Upload `.txt` or paste text directly  
- ✂️ Automatic chunking (1000 chars, 150 overlap)  
- 🧩 Embeddings with **Nomic v1.5 (768-dim)**  
- 🗄️ Vector storage in **Supabase pgvector**  
- 🔍 Retrieval (Top-K) + **Cohere Rerank**  
- 🤖 Answer generation with **Groq LLaMA-3 (llama3-8b-8192)**  
- 🔖 Inline bracket citations `[1], [2]`  
- 🧹 Clear Index button to reset documents  

---

## 🏗️ Architecture

```
[User] --(text/query)--> [Streamlit Frontend]
         |                       |
         | upload/paste          | chunk (1000/150)
         v                       v
    [Nomic Embed v1.5]
         | (768-d)
         v
    [Supabase pgvector]
    (table: documents)
         ^
         | retrieve top-k (RPC)
         v
    [Cohere Rerank]
         v
    context blocks
         v
    [Groq LLaMA-3 LLM]
         v
[Answer + [1],[2]] <--- Sources with metadata
```

---

## 🔐 Environment Variables

Create a `.env` file (or add as secrets in hosting platform):

```bash
NOMIC_API_KEY=your_nomic_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
```

⚠️ **Keep your Supabase SERVICE_ROLE_KEY server-side only.**

---

## 🗄️ Supabase Setup (pgvector)

```sql
create extension if not exists vector;

create table if not exists public.documents (
  id bigserial primary key,
  content text not null,
  metadata jsonb,
  embedding vector(768)
);

create index if not exists documents_embedding_ivfflat
on public.documents using ivfflat (embedding vector_l2_ops)
with (lists = 100);

create or replace function public.match_documents(
  query_embedding vector(768),
  match_count int default 5
)
returns table(
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language sql stable as $$
  select d.id, d.content, d.metadata,
         1 - (d.embedding <-> query_embedding) as similarity
  from public.documents d
  order by d.embedding <-> query_embedding
  limit match_count;
$$;
```

---

## 🧩 Chunking & Metadata

- **Chunk size**: 1000 characters
- **Overlap**: 150 characters
- **Metadata stored**: `{ "source": filename, "position": chunk_index }`

---

## 🔎 Retrieval & Reranking

- **Retrieval**: Top-K (k=5) via RPC
- **Reranker**: Cohere rerank-english-v3.0, top_n=3
- **Fallback**: uses vector order if reranker unavailable

---

## 🧠 LLM & Answering

- **Model**: Groq LLaMA-3 (llama3-8b-8192)
- **Citations**: Answers include inline citations `[i]`
- **No context**: responds "cannot answer from provided docs"

---

## 🖥️ Frontend (Streamlit)

- Upload/paste text area
- Query box → triggers pipeline
- Answer panel with citations
- Expandable sources panel
- Clear Index button

---

## 🚀 Deployment

### Hugging Face Spaces (Streamlit)
1. Create new Space → choose Streamlit
2. Add repo files
3. Add secrets in Space Settings

**Example requirements.txt:**
```
streamlit
supabase
langchain
nomic
cohere
groq
```

### Render (alternative)
- **Build command**: `pip install -r requirements.txt`
- **Start command**: 
  ```bash
  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
  ```

---

## ⚖️ Limits & Tradeoffs

- Free API tiers may throttle requests
- Reranking improves quality but adds latency
- Larger Top-K → better recall but noisier context
- Token usage/costs estimated approximately

---

## 📂 Project Structure

```
.
├── app.py               # Streamlit app
├── requirements.txt
├── README.md
├── .env.example
└── sql/
    └── setup.sql        # Supabase pgvector setup
```

---

## 📄 Documentation

- Quick-start and setup are here in README.md
- Full detailed report available as PDF in `/docs` (if included)

---

## 📝 License

MIT (or as per your choice)
