# 🩺 Medical RAG QA Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-rag-app-assistant-taqi-tallal.streamlit.app/)

## 📖 Overview
This repository implements a Retrieval-Augmented Generation (RAG) system that answers medical questions using a local vector store of medical transcriptions. The assistant retrieves relevant clinical context and generates answers grounded in the retrieved sources. Every answer includes citation information (source document and specialty) to improve transparency and reduce hallucinations.

## 🌟 Key Features
- Evidence-based answers grounded in the dataset
- Citation-aware responses (shows "Source Document" and "Specialty")
- RAG pipeline using FAISS for vector search and LangChain for orchestration
- Streamlit-based interactive UI for demos

## 🔗 Live Demo
Try the deployed app here: 
https://medical-rag-app-assistant-taqi-tallal.streamlit.app/  
(If the app is sleeping on free hosting, allow a minute to wake.)

## 🛠️ Tech Stack
- LLM: Google Gemini Pro (`gemini-pro`)
- Embeddings: Google Generative AI Embeddings (`models/text-embedding-004`)
- Orchestration: LangChain (v0.3+)
- Vector DB: FAISS (local, CPU)
- Interface: Streamlit
- Language: Python 3.10+

## 📂 Dataset
- Source: Medical Transcriptions Dataset (Kaggle) — https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
- Content: Medical transcription samples across specialties
- Preprocessing: cleaned, chunked (≈1000 characters), embedded and stored in a FAISS index

## ⚙️ Installation & Local Setup

1. Clone the repository
```bash
git clone https://github.com/SyedTaqii/medical-rag-qa-assistant.git
cd medical-rag-qa-assistant
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add API key(s)
Create a `.env` file in the project root with your Google API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```
(Use Google AI Studio to obtain credentials as required by your LLM/embedding setup.)

5. Run the app
```bash
streamlit run app.py
```

## 📊 Evaluation
The system was evaluated with 30 medical queries spanning multiple specialties (Cardiology, ENT, Surgery, etc.). Results and test cases are in `task1_evaluation.csv`. Evaluation focused on:
- Faithfulness (whether answers used retrieved context)
- Relevance and correctness of the response

## 📁 Project Structure
```
├── app.py                  # Streamlit application (main)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── vectorstore/            # FAISS index files
│   ├── index.faiss
│   └── index.pkl
└── task1_evaluation.csv    # Evaluation results / test queries
```

## ⚖️ License & Data
The dataset used (tboyle10) is provided under CC0 (public domain) on Kaggle. Verify dataset license before commercial use.