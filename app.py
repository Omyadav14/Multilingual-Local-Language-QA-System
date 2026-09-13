import re
import urllib.request
import urllib.error
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

APP_DIR = Path(__file__).parent
DATA_FILE = APP_DIR / "multilingual_qa_dataset.csv"

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
}

SCRIPT_RANGES = {
    "hi": r"[\u0900-\u097F]",
    "mr": r"[\u0900-\u097F]",
    "bn": r"[\u0980-\u09FF]",
    "gu": r"[\u0A80-\u0AFF]",
    "ta": r"[\u0B80-\u0BFF]",
    "te": r"[\u0C00-\u0C7F]",
}

# Common Devanagari words help distinguish Hindi and Marathi in this prototype.
MARATHI_HINTS = {"आहे", "काय", "कोणती", "म्हणजे", "ही", "चा", "ची", "चे"}
HINDI_HINTS = {"है", "क्या", "कौन", "कौनसा", "कौन सा", "का", "की", "के"}

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error("Dataset file not found: multilingual_qa_dataset.csv")
        st.stop()
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_model():
    # Multilingual MiniLM is lightweight enough for a college prototype.
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def detect_language(text):
    text = text.strip()
    for code in ["bn", "gu", "ta", "te"]:
        if re.search(SCRIPT_RANGES[code], text):
            return code

    if re.search(SCRIPT_RANGES["hi"], text):
        words = set(re.findall(r"[\u0900-\u097F]+", text))
        if words & MARATHI_HINTS:
            return "mr"
        if words & HINDI_HINTS:
            return "hi"
        # Devanagari-only input is ambiguous; default to Hindi for this prototype.
        return "hi"

    return None

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

def build_index(model, df):
    embeddings = model.encode(
        df["question"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False
    )
    vectors = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def retrieve(question, model, index, df, language=None, top_k=3):
    query_vector = model.encode([question], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")

    # Retrieve a wider set first, then prefer the detected language.
    scores, ids = index.search(query_vector, min(top_k * 4, len(df)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = df.iloc[int(idx)].to_dict()
        row["score"] = float(score)
        results.append(row)

    if language:
        preferred = [r for r in results if r["language"] == LANGUAGE_NAMES[language]]
        others = [r for r in results if r["language"] != LANGUAGE_NAMES[language]]
        results = preferred + others

    return results[:top_k]

def ask_ollama(question, language_name, contexts, model_name="qwen2.5:3b"):
    context_text = "\n\n".join(
        f"Source: {item['source']}\nInformation: {item['answer']}"
        for item in contexts
    )

    prompt = f"""You are a multilingual question answering assistant.
Answer the user's question in {language_name}.
Use only the supplied context. If the context does not contain enough information,
say that the information is not available in the knowledge base.
Keep the answer concise.

Question: {question}

Context:
{context_text}
"""

    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("response", "").strip(), None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, str(exc)

st.set_page_config(
    page_title="Multilingual Local Language QA",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Multilingual Local Language Question Answering System")
st.caption("NLP + Multilingual Embeddings + FAISS + RAG + Open-Source LLM")

with st.sidebar:
    st.header("⚙️ Settings")
    use_ollama = st.checkbox("Use local Qwen2.5-3B-Instruct via Ollama", value=False)
    top_k = st.slider("Number of retrieved results", 1, 5, 3)
    st.info("Supported: Hindi, Marathi, Bengali, Gujarati, Tamil and Telugu.")

df = load_data()
model = load_model()
index = build_index(model, df)

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_area(
    "Ask your question",
    placeholder="Example: भारताची राजधानी कोणती आहे?",
    height=110
)

if st.button("🔎 Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    cleaned = clean_text(question)
    lang_code = detect_language(cleaned)

    if lang_code is None:
        st.error("Unsupported language. Please use Hindi, Marathi, Bengali, Gujarati, Tamil or Telugu.")
        st.stop()

    language_name = LANGUAGE_NAMES[lang_code]
    results = retrieve(cleaned, model, index, df, lang_code, top_k)

    st.success(f"Detected language: {language_name}")

    answer = None
    ollama_error = None

    if use_ollama:
        answer, ollama_error = ask_ollama(cleaned, language_name, results)

    if answer:
        final_answer = answer
    else:
        # Safe prototype fallback: return the most relevant knowledge-base answer.
        final_answer = results[0]["answer"]
        if use_ollama and ollama_error:
            st.caption("Ollama was unavailable, so the most relevant knowledge-base answer is shown.")

    st.subheader("💬 Answer")
    st.write(final_answer)

    st.subheader("📚 Sources / Retrieved Context")
    for i, item in enumerate(results, start=1):
        with st.expander(f"{i}. {item['source']} — similarity {item['score']:.3f}"):
            st.write(f"**Question:** {item['question']}")
            st.write(f"**Information:** {item['answer']}")

    st.session_state.history.append({
        "question": cleaned,
        "language": language_name,
        "answer": final_answer
    })

if st.session_state.history:
    st.divider()
    st.subheader("📝 Conversation History")
    for item in reversed(st.session_state.history[-10:]):
        st.markdown(f"**{item['language']} — Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")
        st.divider()

st.caption("Academic prototype • Local/open-source AI approach • Knowledge-base-grounded answering")
