# Production AI & Log Analytics Console

An enterprise-ready, containerized system demonstrating machine learning pipelines, high-performance streaming Python engineering, and Retrieval-Augmented Generation (RAG) architectures.

## 🎥 Video Demonstrations

- **[Demo 1: Machine Learning Pipeline & Antigravity IDE](https://youtu.be/bI9uKCWMQLY)**
- **[Demo 2: Log Analyzer API](https://youtu.be/RIXwZdb_c20)**
- **[Demo 3: RAG Application & Web Console](https://youtu.be/v78zcm76VCI)**

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or 3.13 (virtual environment recommended)
- [Docker & Docker Compose](https://docs.docker.com/get-docker/) (for containerized deployment)

### Local Development Setup
1. **Clone and Navigate**:
   ```bash
   cd AI_ML_Assessment
   ```

2. **Activate Virtual Environment**:
   * Windows PowerShell:
     ```powershell
     .venv\Scripts\activate
     ```
   * Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Run Problem 1 (Machine Learning)**:
   ```bash
   # 1. Generate the synthetic dataset
   python problem_1_ml/generate_data.py
   
   # 2. Run training, threshold sweeps, and evaluation
   python problem_1_ml/train.py
   ```
   *Plots comparing PR/ROC curves and confusion matrices will be saved to `problem_1_ml/plots/`.*

5. **Run Problem 2 (Log Analyzer API)**:
   *Open a **new terminal tab**, activate the virtual environment (`.venv\Scripts\activate`), and run:*
   ```bash
   # Start the FastAPI backend server on port 8000
   python -m uvicorn problem_2_python.api:app --host 127.0.0.1 --port 8000 --reload
   ```

6. **Run Problem 3 (RAG App & Web Console)**:
   *Open a **third terminal tab**, activate the virtual environment (`.venv\Scripts\activate`), and run:*
   ```bash
   # Start the RAG server & UI client on port 8001
   python -m uvicorn problem_3_rag.app:app --host 127.0.0.1 --port 8001 --reload
   ```
   *Open [http://127.0.0.1:8001](http://127.0.0.1:8001) in your browser to access the interactive dashboard.*

---

## 🐳 Docker Deployment (Recommended)

To launch the complete containerized stack (both the Log Processor Backend and the RAG Web Application), run:

```bash
docker-compose up --build
```

- **Unified Web UI / RAG Console**: [http://localhost:8001](http://localhost:8001)
- **Log Processor API**: [http://localhost:8000](http://localhost:8000)

*Note: If you have a Google Gemini API Key, you can pass it to the containers to enable real Gemini responses:*
```bash
$env:GEMINI_API_KEY="your_api_key_here"; docker-compose up --build
```
*If no key is present, the system seamlessly falls back to a local **Rule-based Contextual Synthesizer** so the app remains fully functional out-of-the-box.*

---

## 📂 Architecture Directory Structure

```
AI_ML_Assessment/
│
├── problem_1_ml/
│   ├── generate_data.py       # Dataset generation script (100k samples, 0.5% failure class)
│   ├── pipeline.py            # Sklearn Pipeline, standard scaler, custom OutlierClipper
│   ├── evaluate.py            # Custom evaluations: PR-AUC, ROC-AUC, threshold sweep cost
│   ├── train.py               # Model fit coordinator (SMOTE+XGBoost vs Cost-Sensitive RF)
│   └── technical_memo.md      # Analysis on metric selection and mathematical threshold tuning
│
├── problem_2_python/
│   ├── log_processor.py       # Memory-efficient log processor using context managers and generators
│   ├── api.py                 # FastAPI endpoint (POST /analyze-logs) streaming uploads in chunks
│   ├── Dockerfile             # Containerizes the Log analyzer API on port 8000
│   └── tests/
│       └── test_log_processor.py  # Pytest suite for log processing validation
│
├── problem_3_rag/
│   ├── config.py              # Configuration paths and hyper-parameters
│   ├── mock_data/
│   │   └── hr_policy.txt      # Mock 2-page Acme Corp HR policy manual
│   ├── rag_pipeline.py        # LangChain, ChromaDB, HuggingFace local embeddings
│   ├── app.py                 # FastAPI server serving the RAG API and the static UI files
│   ├── Dockerfile             # Containerizes the RAG service and web UI on port 8001
│   └── static/                # Single Page Web App (HTML5 / CSS3 / Vanilla JS)
│
├── docker-compose.yml         # Multi-container orchestration config
└── README.md                  # System documentation & Production Debugging
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **ML Framework** | Scikit-Learn, XGBoost, imbalanced-learn (SMOTE) |
| **Data** | Pandas, NumPy, Matplotlib |
| **Backend API** | FastAPI, Uvicorn |
| **RAG / LLM** | LangChain, Google Gemini API (with offline fallback) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local, no API needed) |
| **Vector Database** | ChromaDB (persistent local storage) |
| **Frontend** | HTML5, CSS3 (Glassmorphism dark theme), Vanilla JavaScript |
| **Containerization** | Docker, Docker Compose |
| **Testing** | Pytest |

---

## 📊 Problem 1 Results: Model Comparison

### Evaluation Metrics

| Metric | SMOTE + XGBoost | Cost-Sensitive Random Forest |
|---|---|---|
| **ROC-AUC** | 0.9264 | 0.9145 |
| **PR-AUC** | 0.6405 | 0.3501 |
| **Precision (Failure)** | 0.65 | 0.37 |
| **Recall (Failure)** | 0.62 | 0.39 |
| **F1-Score (Failure)** | 0.64 | 0.38 |

> **Winner: SMOTE + XGBoost** — nearly 2x the PR-AUC of Random Forest, meaning it catches far more real failures with fewer false alarms.

### Why We Chose PR-AUC Over Accuracy
- **Accuracy is misleading** on imbalanced data. A model that always predicts "no failure" achieves 99.5% accuracy but catches zero failures.
- **ROC-AUC (0.92)** shows strong overall class separation, but can be overly optimistic when negatives dominate.
- **PR-AUC (0.64)** is the gold standard for rare-event detection. It strictly evaluates performance on the minority class (failures), ignoring the easy-to-predict majority.

---

## 🎯 Threshold Optimization (Business Cost Alignment)

Instead of using the default 0.50 probability threshold, we performed a mathematical sweep across hundreds of thresholds to minimize real-world business cost:

| Configuration | FN (Missed Failures) | FP (False Alarms) | Total Cost |
|---|---|---|---|
| Default Threshold (0.50) | 42 | 37 | **$4,237** |
| Optimized Threshold (0.2050) | 25 | 437 | **$2,937** |

**Cost Function:** `Total Cost = (FN x $100) + (FP x $1)`

> By lowering the threshold, we deliberately accept more false alarms ($1 each) to catch more real failures ($100 each). This saved **$1,300 (30%)** in projected maintenance costs.

The threshold sweep plots and confusion matrix comparisons are saved in `problem_1_ml/plots/`.

---

## ⚡ Problem 2: Complexity & Memory Analysis

![Transaction Log Analyzer Console](assets/log_analyzer.png)

### The Problem
The original log processor loaded the **entire file into memory** at once. For a 50GB production log file, this would crash the server with an Out-Of-Memory (OOM) error.

### Our Solution: O(1) Space Streaming

| Aspect | Original Code | Our Implementation |
|---|---|---|
| **Memory** | O(n) — loads entire file | O(1) — streams line-by-line |
| **Approach** | `file.readlines()` | Python Generators (`yield`) |
| **Resource Safety** | Manual close | Context Managers (`__enter__` / `__exit__`) |
| **API Integration** | None | FastAPI `UploadFile` with async chunk streaming |

**Key Design Patterns Used:**
- **Generator Functions (`yield`)**: Each line is processed and immediately discarded, keeping memory flat regardless of file size.
- **Context Managers**: The `LogProcessor` class implements `__enter__` and `__exit__` to guarantee file handles are always closed, even if an exception occurs.
- **Streaming API**: The FastAPI endpoint reads uploaded files in 1MB chunks, never buffering the full file.

---

## 🧠 RAG Architecture (Problem 3)

![Enterprise HR Policy Assistant RAG Console](assets/rag_console.png)

```
User Query ("How much maternity leave do I get?")
        │
        ▼
┌─────────────────────────┐
│   Query Embedding       │  HuggingFace all-MiniLM-L6-v2 (local)
│   (384-dim vector)      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   ChromaDB Vector Store │  Cosine similarity search
│   (Persistent on disk)  │  Returns top-K relevant chunks
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   LLM Generation        │  Google Gemini API (or local fallback)
│   Prompt = Context +    │  "Answer ONLY from the provided context"
│   User Query            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Structured Response   │  Returned to the Web Dashboard
└─────────────────────────┘
```

**Key Design Decisions:**
- **Local Embeddings**: We use HuggingFace's `all-MiniLM-L6-v2` model locally, so no API calls or costs are incurred for embedding generation.
- **Offline Fallback**: If no `GEMINI_API_KEY` is set, the system automatically switches to a **Rule-based Contextual Synthesizer** that parses retrieved chunks using string matching. The app works fully offline.
- **Persistent Vector Store**: ChromaDB stores embeddings on disk, so the HR policy only needs to be indexed once. Subsequent server restarts skip re-indexing.

---

## 🛡️ Hallucination Prevention & Fixes

### The Scenario
*In production, the RAG system starts hallucinating. When a user asks about "Maternity Leave", the retriever fetches chunks related to "Sick Leave", causing the LLM to give the wrong answer confidently.*

### Root Cause Analysis & Fixes

#### 1. Evaluating Retrieval Quality
- **Hit Rate (Recall@K)**: Measures how often the correct chunk appears in the top K results.
- **Mean Reciprocal Rank (MRR)**: Evaluates the position of the first relevant chunk. Target: MRR > 0.85.
- **Cosine Similarity Drift**: If "Maternity Leave" queries are close to "Sick Leave" chunks in vector space, the embedding model is failing.
- **Monitoring Tools**: Integrate **Ragas** or **TruLens** for continuous Context Recall and Context Precision evaluation.

#### 2. Chunking Strategy Optimizations
- **Metadata Filtering**: Tag chunks during ingestion (e.g., `{"category": "maternity_leave"}`). Route queries to the correct category before vector search, eliminating cross-topic contamination.
- **Parent-Document Retrieval**: Use small chunks (100 tokens) for precise matching, but retrieve the larger parent chunk (1000 tokens) for LLM context.
- **Semantic Chunking**: Split documents by detecting semantic shifts between sentences instead of fixed character counts.

#### 3. Guardrails & Fallback Mechanisms
- **Similarity Thresholding**: Reject retrieved chunks below a minimum cosine similarity score (e.g., 0.35).
- **LLM-as-a-Judge**: Before generating the final answer, run a validation prompt: *"Is this context relevant to the query? YES/NO"*. If NO, skip generation.
- **Structured Fallback**: When confidence is low, suppress generation and return a safe response:
  > *"I'm sorry, I could not find a reliable policy matching your question. Let me connect you with Acme HR Support."*

---

## 🚀 Future Improvements

| Area | Improvement | Impact |
|---|---|---|
| **ML Pipeline** | Add LightGBM and CatBoost models to the comparison | May improve PR-AUC further on categorical features |
| **ML Pipeline** | Implement cross-validation with stratified K-fold | More robust threshold estimates across data splits |
| **Log Analyzer** | Add real-time WebSocket streaming for live log monitoring | Enable dashboards to show flagged transactions in real-time |
| **Log Analyzer** | Support additional log formats (JSON, CSV, Syslog) | Broader production applicability |
| **RAG System** | Fine-tune embedding model on domain-specific HR vocabulary | Improve retrieval accuracy for niche policy terms |
| **RAG System** | Add multi-document support (upload multiple HR policies) | Scale to enterprise-level document management |
| **RAG System** | Implement conversation memory (chat history context) | Enable follow-up questions like "What about for part-time employees?" |
| **Infrastructure** | Add CI/CD pipeline with GitHub Actions | Automated testing and deployment on every push |
| **Infrastructure** | Add Prometheus + Grafana monitoring for API latency | Production observability and alerting |
| **Security** | Implement API key authentication on all endpoints | Prevent unauthorized access in production |

