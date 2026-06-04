# Production AI & Log Analytics Console

An enterprise-ready, containerized system demonstrating machine learning pipelines, high-performance streaming Python engineering, and Retrieval-Augmented Generation (RAG) architectures.

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

## 📊 Machine Learning Evaluation & Business Value

In this project, predicting rare machine failures requires specialized metrics and business alignment. To address the **0.5% minority class**, SMOTE (Synthetic Minority Over-sampling Technique) and algorithm-level class weighting were evaluated during model training.

- **Why PR-AUC > ROC-AUC:** Although ROC-AUC remained high (0.92), PR-AUC was prioritized because it focuses purely on performance for the minority failure class, making it vastly more informative under severe class imbalance. XGBoost achieved a PR-AUC of **0.64**, significantly outperforming the Cost-Sensitive Random Forest (0.35).
- **Business Cost Optimization:** Missing a failure costs the business $100, while a false alarm costs $1. By sweeping probability thresholds mathematically, we optimized the threshold down from the default 0.50 to **0.2050**.
- **Recall Optimization:** Predictive maintenance relies heavily on Recall. The optimized threshold increased Recall from **62.5% to 77.6%**, ensuring more critical machine failures were detected before causing downtime. This alignment dropped the total business cost from $4,237 (default threshold) to **$2,937**, saving the company over 30%!
- **Production Perspective:** In production, this threshold should be monitored and recalibrated periodically as failure patterns and financial business costs evolve over time.

---

## 🛠️ Production Debugging (Crucial Scenario)

### Scenario
*In production, the RAG system starts hallucinating. When a user asks about **"Maternity Leave"**, the retriever fetches chunks related to **"Sick Leave"**, causing the LLM to give the wrong answer confidently.*

### Step-by-Step Mitigation & Debugging Strategy

#### 1. Evaluating Retrieval Quality
To diagnose if the issue is a retrieval failure or generation failure, we implement structured retrieval evaluations using a test set of user queries mapped to target document chunks:
- **Hit Rate (Recall@K)**: Measures the percentage of times the correct policy chunk is returned in the top $K$ results. If this is low for "Maternity Leave", the retriever is failing.
- **Mean Reciprocal Rank (MRR)**: Evaluates the position of the first relevant chunk. If the relevant chunk is at rank 4, the score is $1/4 = 0.25$. We aim for $\text{MRR} > 0.85$ in production.
- **Cosine Similarity Drift**: Compare query embedding cosine similarities. If "Maternity Leave" queries are physically close to "Sick Leave" chunks in vector space, the embedding model is failing to partition the topics.
- **Frameworks**: Integrate automated tools like **Ragas** or **TruLens** to measure **Context Recall** and **Context Precision** continuously on logged production queries.

#### 2. Chunking Strategy Optimizations
Simple character-based chunking often splits sentences mid-thought, causing semantic loss. We apply the following upgrades:
- **Metadata Filtering (Query Routing)**: Append tag fields to document chunks during ingestion (e.g. `{"category": "maternity_leave"}`). Implement a lightweight router model or intent classifier at the query layer. If the query is about maternity, apply a metadata filter `where category == 'maternity_leave'` to ChromaDB, completely eliminating the possibility of fetching "Sick Leave" chunks.
- **Parent-Document Retrieval**: Chunk documents into small sizes (e.g., 100 tokens) for highly granular vector search matching, but store them linked to a larger parent chunk (e.g., 1000 tokens). When a small chunk matches, retrieve and pass the parent chunk to the LLM. This provides broad context to prevent truncation errors.
- **Semantic Chunking**: Instead of fixed token counts, chunk the document by detecting semantic shifts (embedding differences between sentences). This ensures each chunk represents a single cohesive concept.

#### 3. Fallback Mechanisms & Guardrails
To prevent the LLM from confidently answering using irrelevant chunks, we enforce the following safeguards:
- **Similarity Thresholding**: Establish a strict minimum similarity score (e.g. cosine distance $\le 0.35$). If retrieved chunks fall below this threshold, the retriever flags a low-confidence state.
- **Context Relevance Guardrail (LLM-as-a-Judge)**: Before generating the final response, run a rapid validation prompt on the retrieved chunks (e.g., *"Is this document context relevant to the user query? Return YES or NO"*). If it returns "NO", skip the generation.
- **Structured Fallback Response**: When low confidence or irrelevant context is flagged, suppress generation and output a safe, pre-defined response:
  > *"I'm sorry, I could not find a reliable policy matching your question in my database. Let me connect you directly with the Acme HR Support team."*
