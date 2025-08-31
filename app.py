
import os
import re
import json
from typing import List, Dict, Any

import streamlit as st
from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings (Nomic)
from nomic import embed
import nomic  # for login

# Reranker (Cohere)
import cohere

# LLM (Groq - LLaMA 3)
from groq import Groq


# ===============================
# Environment (set these as HF Spaces Secrets)
# ===============================
NOMIC_API_KEY = os.environ.get("NOMIC_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Basic validation
missing = [k for k, v in {
    "NOMIC_API_KEY": NOMIC_API_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    "COHERE_API_KEY": COHERE_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
}.items() if not v]

# ===============================
# Initialize clients
# ===============================
supabase: Client = None
co: cohere.Client | None = None
groq_client: Groq | None = None

def init_clients():
    global supabase, co, groq_client
    if NOMIC_API_KEY:
        try:
            nomic.login(token=NOMIC_API_KEY)  # IMPORTANT: use token=
        except Exception as e:
            st.error(f"Nomic login failed: {e}")

    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        except Exception as e:
            st.error(f"Supabase client init failed: {e}")

    if COHERE_API_KEY:
        try:
            co = cohere.Client(COHERE_API_KEY)
        except Exception as e:
            st.error(f"Cohere client init failed: {e}")

    if GROQ_API_KEY:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            st.error(f"Groq client init failed: {e}")

init_clients()


# ===============================
# RAG helper functions
# ===============================
EMBED_DIM = 768   # nomic-embed-text-v1.5 output dimension
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    res = embed.text(texts=texts, model="nomic-embed-text-v1.5")
    return res["embeddings"]

def upsert_documents(chunks: List[str], source_name: str) -> int:
    """
    Insert chunks into Supabase.
    No auto-clear: lets you add multiple docs per session.
    Use the Clear Index button in the sidebar to reset.
    """
    if not supabase:
        st.error("Supabase is not initialized.")
        return 0

    try:
        embs = embed_texts(chunks)
        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embs)):
            rows.append({
                "content": chunk,
                "metadata": {"position": i, "source": source_name},
                "embedding": emb,
            })

        if rows:
            supabase.table("documents").insert(rows).execute()
        return len(rows)
    except Exception as e:
        st.error(f"Embedding/Insert error: {e}")
        return 0

def retrieve_top_k(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not supabase:
        st.error("Supabase is not initialized.")
        return []
    try:
        q_emb = embed_texts([query])[0]
        resp = supabase.rpc("match_documents", {
            "query_embedding": q_emb,
            "match_count": k
        }).execute()
        return resp.data or []
    except Exception as e:
        st.error(f"Retrieval error (RPC match_documents): {e}")
        return []

def rerank_docs(query: str, docs: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    if not docs:
        return []
    if not co:
        return docs[:top_n]

    try:
        texts = [d["content"] for d in docs]
        rr = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=texts,
            top_n=min(top_n, len(texts))
        )
        return [docs[r.index] for r in rr.results]
    except Exception as e:
        st.warning(f"Rerank failed, using vector order. Error: {e}")
        return docs[:top_n]

def call_llm(context_blocks: List[str], query: str) -> str:
    if not groq_client:
        return "Error: LLM not initialized."

    ctx = "\n\n---\n\n".join([f"[{i+1}]\n{blk}" for i, blk in enumerate(context_blocks)])
    prompt = f"""You are a grounded QA assistant. Use ONLY the context below to answer.
Context:
{ctx}
Rules:
- Always cite supporting claims with inline bracket citations like [1], [2].
- If the answer is not explicitly stated, you may infer or rephrase *why/how* answers as long as it logically follows from the context.
- If nothing relevant is found at all, then say you cannot answer from the provided documents.
- Stay faithful to the context (no hallucinations).
- Be clear, concise, and helpful.
Question: {query}
Answer with citations:"""

    try:
        chat = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        return chat.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {e}"

def extract_cited_indices(answer: str) -> List[int]:
    return sorted({int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer)})


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="RAG (Supabase + Nomic + Cohere + Groq)", layout="wide")
st.title("Chat with your Documents")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload .txt (you can upload multiple one by one)", type=["txt"])
    pasted = st.text_area("Or paste text", height=240, placeholder="Paste your document text here…")
    source_name = "uploaded.txt" if uploaded else "pasted.txt" if pasted else "untitled.txt"

    st.divider()
    st.subheader("Status")
    if missing:
        st.error(f"Missing secrets: {', '.join(missing)}")
    else:
        st.success("All secrets present")

    if st.button("Clear Index (Supabase)"):
        if supabase:
            try:
                supabase.table("documents").delete().neq("id", -1).execute()
                st.success("Cleared documents table.")
            except Exception as e:
                st.error(f"Failed to clear table: {e}")

# Load text
doc_text = ""
if uploaded is not None:
    try:
        doc_text = uploaded.read().decode("utf-8", errors="ignore")
        st.success(f"File {uploaded.name} uploaded.")
    except Exception as e:
        st.error(f"Read error: {e}")
elif pasted.strip():
    doc_text = pasted.strip()

# ===============================
# Query Input (NO form, runs directly)
# ===============================
query = st.text_input("Ask a question about your documents:")

# Run query as soon as user types (if not empty)
if query.strip():
    # 1) If new doc provided, index it
    if doc_text:
        chunks = chunk_text(doc_text)
        n = upsert_documents(chunks, source_name)
        if n > 0:
            st.success(f"Indexed {n} chunks from {source_name}")

    # 2) Retrieval + rerank + LLM
    retrieved = retrieve_top_k(query, k=5)
    reranked = rerank_docs(query, retrieved, top_n=3)
    ctx_blocks = [d["content"] for d in reranked]

    answer = call_llm(ctx_blocks, query)

    st.subheader("Answer")
    st.markdown(answer)

    cited = extract_cited_indices(answer)
    show_docs = []
    fallback = False
    if cited:
        for i in cited:
            if 1 <= i <= len(ctx_blocks):
                show_docs.append(reranked[i-1])
    if not show_docs:
        show_docs = reranked
        fallback = True

    if show_docs:
        if fallback:
            st.warning("No citations detected in the answer. Showing all retrieved sources used as context.")
        st.subheader("Sources")
        for i, d in enumerate(show_docs, start=1):
            meta = d.get("metadata") or {}
            src = meta.get("source", source_name)
            pos = meta.get("position", "?")
            with st.expander(f"Source {i} — {src} (chunk #{pos})"):
                st.write(d["content"])
else:
    st.info("Type a question above to run query.")
