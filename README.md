# 🧾 GenAI Tax Optimization Assistant

An end-to-end AI-powered tax advisor using **RAG + LLM** (Claude), **FastAPI** backend, and a polished **HTML/CSS/JS** frontend.

---

## Architecture

```
┌─────────────────────────────────┐
│        Frontend (HTML/CSS/JS)   │
│  - Financial data input form    │
│  - AI recommendations viewer    │
│  - Real-time chat assistant     │
└────────────┬────────────────────┘
             │ REST API (JSON)
┌────────────▼────────────────────┐
│        FastAPI Backend          │
│  /api/analyze  → RAG + LLM      │
│  /api/chat     → Conversational │
│  /api/sample-profiles           │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│    RAG Pipeline                 │
│  1. Retrieve similar profiles   │
│     (in-memory vector store)    │
│  2. Build context-rich prompt   │
│  3. Call Claude (Anthropic API) │
│  4. Parse & return JSON recs    │
└─────────────────────────────────┘
```

---

## Project Structure

```
tax_optimizer/
├── backend/
│   ├── main.py           ← FastAPI app (RAG + LLM logic)
│   └── requirements.txt
└── frontend/
    └── index.html        ← Single-file HTML/CSS/JS app
```

---

## Setup & Run

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Start the FastAPI server

```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the frontend

Open `frontend/index.html` in your browser — or serve it:

```bash
cd frontend
python -m http.server 3000
# then visit http://localhost:3000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze` | Accepts financial data, runs RAG + Claude, returns JSON recommendations |
| POST | `/api/chat` | Conversational follow-up with optional financial context |
| GET  | `/api/sample-profiles` | Returns sample user financial profiles |

### Sample `/api/analyze` payload

```json
{
  "income": 85000,
  "expenses": 24000,
  "health_insurance": 2400,
  "home_loan": 9000,
  "elss": 0,
  "nps": 3000,
  "ppf": 2000,
  "house_rent": 14400,
  "previous_tax_amount": 12000,
  "tax_credits": 2000,
  "state": "CA",
  "filing_status": "Single"
}
```

### Sample response

```json
{
  "success": true,
  "data": {
    "estimated_tax_liability": 18500,
    "potential_savings": 4200,
    "effective_tax_rate": 21.8,
    "recommendations": [
      {
        "category": "Retirement",
        "strategy": "Maximize 401(k) Contributions",
        "description": "...",
        "potential_savings": 1800,
        "priority": "High"
      }
    ],
    "summary": "...",
    "deduction_breakdown": {
      "health_insurance": 2400,
      "home_loan_interest": 9000,
      "retirement_contributions": 3000,
      "house_rent": 14400,
      "other": 0
    }
  }
}
```

---

## How RAG Works Here

1. **Retrieval**: User's income and filing status are used to find the 3 most similar profiles from the in-memory store (simulating ChromaDB similarity search from the notebook).
2. **Augmentation**: Retrieved profiles + a tax knowledge base are appended to the prompt as context.
3. **Generation**: Claude generates structured JSON recommendations tailored to the user's specific financial situation.

> For production, replace the in-memory store with ChromaDB + sentence-transformers embeddings exactly as shown in the original notebook.

---

## From the Notebook

The notebook (`LLM___RAG_for_Finance_V4.ipynb`) uses:
- `sentence-transformers` for embeddings
- `ChromaDB` as vector store
- `Mistral-7B` via HuggingFace pipeline as LLM
- `LangChain RetrievalQA` chain

This project replaces the HuggingFace LLM with Claude (better quality, no GPU needed) while keeping the same RAG pattern and prompt template structure.
