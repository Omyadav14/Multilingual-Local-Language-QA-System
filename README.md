Website link:-https://multilingual-local-language-app-system-hksetruxnzub37jgxyqj4t.streamlit.app/

# 🌐 Multilingual Local Language Question Answering System

An AI-powered question answering prototype for Indian local languages. The system accepts a question, detects the language, retrieves relevant information from a multilingual knowledge base using semantic search, and can generate a grounded answer using a local LLM through Ollama.

## ✨ Features

- 🌍 Multilingual question input
- 🔎 Automatic language detection
- 🧹 Basic NLP preprocessing
- 🧠 Multilingual Sentence Transformer embeddings
- ⚡ FAISS semantic retrieval
- 📚 Knowledge-base retrieval
- 🤖 Optional RAG answer generation with Qwen2.5-3B-Instruct through Ollama
- 💬 Same-language responses
- 📖 Source/reference display
- 📝 Conversation history during the session
- 🚫 Unsupported-language handling
- 🖥️ Streamlit web interface

## 🌐 Initial Supported Languages

- Hindi
- Marathi
- Bengali
- Gujarati
- Tamil
- Telugu

## 🧠 Workflow

```text
User Question
      ↓
Language Detection
      ↓
Text Preprocessing
      ↓
Multilingual Embedding
      ↓
FAISS Semantic Search
      ↓
Retrieve Relevant Context
      ↓
RAG Prompt
      ↓
Qwen2.5-3B-Instruct via Ollama
      ↓
Same-Language Answer + Source
```

## 🛠️ Tech Stack

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Pandas
- NumPy
- PyTorch
- Ollama
- Qwen2.5-3B-Instruct

## 📂 Project Structure

```text
Multilingual_Local_Language_QA_System/
├── app.py
├── multilingual_qa_dataset.csv
├── requirements.txt
└── README.md
```

## ▶️ How to Run

### 1. Install Python

Python 3.x is recommended.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The application will open in the browser using the local Streamlit URL shown in the terminal.

### 4. Optional: Enable local LLM RAG

Install Ollama and download the model:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running. If Ollama is unavailable, the application still provides the most relevant knowledge-base answer and source instead of failing.

## 📊 Dataset

The included CSV is a small demonstration knowledge base covering the six planned languages. It is intended for the working prototype.

For the full project, the knowledge base can be expanded toward the documented target of approximately 2,000–5,000 multilingual questions/documents using suitable open-source resources and a curated dataset.

## 🔌 Main Application

The prototype keeps the core pipeline in `app.py` so it is easy to demonstrate and submit for a college project. The code can later be split into the modular architecture described in the project documentation.

## 🎯 Project Objectives

- Make information accessible in Indian local languages
- Automatically identify the user's language
- Retrieve relevant information using semantic similarity
- Use RAG to ground generated answers in a knowledge base
- Generate answers in the same language as the question
- Use free/open-source technologies and locally deployable models
- Keep the system extensible to additional Indian languages

## 🚀 Future Improvements

- FastAPI backend
- SQLite conversation storage
- Larger multilingual knowledge base
- More Indian languages
- Advanced evaluation metrics
- Voice input and speech-to-text
- OCR and document QA
- Mobile application
- Domain-specific fine-tuning
- Real-time information retrieval

## 👨‍💻 Author

**Om Yadav**  
B.Sc. Data Science

## 📄 License

This project is developed for academic purposes.
