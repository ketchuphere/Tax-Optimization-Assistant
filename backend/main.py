"""
GenAI Tax Optimization Assistant - FastAPI Backend
Uses Claude API (Anthropic) as LLM, with in-memory RAG simulation via ChromaDB-style retrieval.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
import json
import re

app = FastAPI(title="GenAI Tax Optimization API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# In-memory "vector store" – sample financial records that act as RAG context  #
# --------------------------------------------------------------------------- #
FINANCIAL_RECORDS = [
    {
        "User_ID": 200, "Income": 40241.70, "Expenses": 36791.41,
        "HealthInsurance": 1694.30, "HomeLoan": 8038.15, "ELSS": 0.0,
        "NPS": 135.53, "PPF": 0.0, "HouseRent": 0.0,
        "Previous_Tax_Amount": 16913.54, "State": "ID",
        "Filing_Status": "Head of Household",
        "Tax_Credits": 492.47, "Estimated_Tax": 4042.22,
    },
    {
        "User_ID": 963, "Income": 122192.81, "Expenses": 41288.16,
        "HealthInsurance": 2995.22, "HomeLoan": 0.0, "ELSS": 0.0,
        "NPS": 0.0, "PPF": 3385.47, "HouseRent": 0.0,
        "Previous_Tax_Amount": 15947.90, "State": "ID",
        "Filing_Status": "Head of Household",
        "Tax_Credits": 2915.98, "Estimated_Tax": 24914.91,
    },
    {
        "User_ID": 97, "Income": 117114.68, "Expenses": 48807.00,
        "HealthInsurance": 2301.79, "HomeLoan": 9411.21, "ELSS": 0.0,
        "NPS": 0.0, "PPF": 1217.21, "HouseRent": 0.0,
        "Previous_Tax_Amount": 14191.47, "State": "ID",
        "Filing_Status": "Head of Household",
        "Tax_Credits": 3693.89, "Estimated_Tax": 22124.27,
    },
    {
        "User_ID": 141, "Income": 142855.03, "Expenses": 44088.04,
        "HealthInsurance": 0.0, "HomeLoan": 0.0, "ELSS": 3378.31,
        "NPS": 4895.16, "PPF": 4598.96, "HouseRent": 0.0,
        "Previous_Tax_Amount": 14783.12, "State": "ID",
        "Filing_Status": "Head of Household",
        "Tax_Credits": 1673.68, "Estimated_Tax": 28315.82,
    },
    {
        "User_ID": 450, "Income": 122678.21, "Expenses": 33491.81,
        "HealthInsurance": 330.05, "HomeLoan": 0.0, "ELSS": 3993.28,
        "NPS": 678.31, "PPF": 2210.33, "HouseRent": 8447.24,
        "Previous_Tax_Amount": 3722.42, "State": "IN",
        "Filing_Status": "Head of Household",
        "Tax_Credits": 4772.24, "Estimated_Tax": 22804.56,
    },
    {
        "User_ID": 317, "Income": 65185.29, "Expenses": 6770.46,
        "HealthInsurance": 1921.03, "HomeLoan": 0.0, "ELSS": 0.0,
        "NPS": 1767.37, "PPF": 1927.76, "HouseRent": 3657.13,
        "Previous_Tax_Amount": 15957.37, "State": "VI",
        "Filing_Status": "Head of Household",
        "Tax_Credits": 2990.91, "Estimated_Tax": 9660.64,
    },
]

# TAX regulation knowledge base
TAX_KNOWLEDGE = """
US Tax Regulations & Optimization Strategies:
1. Standard Deduction 2024: Single $14,600 | MFJ $29,200 | HOH $21,900
2. Section 80C equivalents (US): 401k up to $23,000; IRA up to $7,000
3. Health Savings Account (HSA): Deductible up to $4,150 (individual) / $8,300 (family)
4. Home Mortgage Interest Deduction: Interest on loans up to $750,000
5. Child Tax Credit: Up to $2,000 per qualifying child
6. Earned Income Tax Credit (EITC): Based on income and dependents
7. Capital Gains Tax: 0%, 15%, or 20% based on income bracket
8. Self-Employment Tax Deduction: Deduct half of SE tax
9. Student Loan Interest: Up to $2,500 deduction
10. State & Local Tax (SALT) Deduction: Capped at $10,000
11. Charitable Contributions: Deductible up to 60% of AGI
12. NPS/PPF equivalents: Retirement account contributions reduce taxable income
13. House Rent Allowance: HRA exemption based on rent paid
14. ELSS (Equity Linked Savings Scheme): Tax-saving mutual fund with 3-year lock-in
"""


# --------------------------------------------------------------------------- #
# Simple similarity retrieval (cosine-like keyword match as RAG simulation)    #
# --------------------------------------------------------------------------- #
def retrieve_similar_records(user_income: float, filing_status: str, k: int = 3):
    """Return k records closest in income and matching filing status."""
    scored = []
    for rec in FINANCIAL_RECORDS:
        income_sim = 1 / (1 + abs(rec["Income"] - user_income) / 50000)
        status_bonus = 0.3 if rec["Filing_Status"] == filing_status else 0
        scored.append((income_sim + status_bonus, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]


# --------------------------------------------------------------------------- #
# Pydantic models                                                               #
# --------------------------------------------------------------------------- #
class UserFinancialData(BaseModel):
    user_id: Optional[int] = None
    income: float
    expenses: float
    health_insurance: float = 0.0
    home_loan: float = 0.0
    elss: float = 0.0
    nps: float = 0.0
    ppf: float = 0.0
    house_rent: float = 0.0
    previous_tax_amount: float = 0.0
    state: str = "CA"
    filing_status: str = "Single"
    tax_credits: float = 0.0
    estimated_tax: Optional[float] = None


class ChatMessage(BaseModel):
    message: str
    financial_context: Optional[UserFinancialData] = None


# --------------------------------------------------------------------------- #
# Core RAG + LLM function                                                       #
# --------------------------------------------------------------------------- #
def build_rag_prompt(user_data: UserFinancialData) -> str:
    similar = retrieve_similar_records(user_data.income, user_data.filing_status)
    context_records = "\n".join([
        f"User_ID: {r['User_ID']}, Income: {r['Income']}, Expenses: {r['Expenses']}, "
        f"HealthInsurance: {r['HealthInsurance']}, HomeLoan: {r['HomeLoan']}, "
        f"ELSS: {r['ELSS']}, NPS: {r['NPS']}, PPF: {r['PPF']}, "
        f"HouseRent: {r['HouseRent']}, Estimated_Tax: {r['Estimated_Tax']}, "
        f"Filing_Status: {r['Filing_Status']}, Tax_Credits: {r['Tax_Credits']}"
        for r in similar
    ])

    user_query = (
        f"User_ID: {user_data.user_id or 'N/A'}, "
        f"Income: {user_data.income}, Expenses: {user_data.expenses}, "
        f"HealthInsurance: {user_data.health_insurance}, HomeLoan: {user_data.home_loan}, "
        f"ELSS: {user_data.elss}, NPS: {user_data.nps}, PPF: {user_data.ppf}, "
        f"HouseRent: {user_data.house_rent}, "
        f"Previous_Tax_Amount: {user_data.previous_tax_amount}, "
        f"State: {user_data.state}, Filing_Status: {user_data.filing_status}, "
        f"Tax_Credits: {user_data.tax_credits}"
    )

    return f"""
You are an expert AI Tax Optimization Assistant. Using the financial data, similar user profiles (RAG context), and tax regulations below, provide detailed, personalized tax-saving recommendations.

## User Financial Data (to analyze):
{user_query}

## Similar User Profiles (RAG Retrieved Context):
{context_records}

## Tax Regulations & Knowledge Base:
{TAX_KNOWLEDGE}

## Instructions:
Respond ONLY with a valid JSON object (no markdown, no preamble) with this exact structure:
{{
  "estimated_tax_liability": <number>,
  "potential_savings": <number>,
  "effective_tax_rate": <number>,
  "recommendations": [
    {{
      "category": "<category name>",
      "strategy": "<strategy title>",
      "description": "<detailed explanation>",
      "potential_savings": <number>,
      "priority": "High|Medium|Low"
    }}
  ],
  "summary": "<2-3 sentence executive summary>",
  "deduction_breakdown": {{
    "health_insurance": <number>,
    "home_loan_interest": <number>,
    "retirement_contributions": <number>,
    "house_rent": <number>,
    "other": <number>
  }}
}}
"""


async def call_claude(prompt: str, system: str = "") -> str:
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=system or "You are a precise tax optimization AI. Always respond with valid JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# --------------------------------------------------------------------------- #
# Routes                                                                        #
# --------------------------------------------------------------------------- #
@app.get("/")
async def root():
    return {"status": "GenAI Tax Optimization API is running", "version": "1.0.0"}


@app.post("/api/analyze")
async def analyze_taxes(user_data: UserFinancialData):
    """Main endpoint: accepts financial data, runs RAG + LLM, returns recommendations."""
    try:
        prompt = build_rag_prompt(user_data)
        raw = await call_claude(prompt)

        # Strip any accidental markdown fences
        clean = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)
        return JSONResponse(content={"success": True, "data": result})

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM returned malformed JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(msg: ChatMessage):
    """Conversational endpoint for follow-up tax questions."""
    try:
        system = (
            "You are a friendly, knowledgeable tax optimization assistant. "
            "Answer concisely and accurately. Use plain text, no JSON needed."
        )
        context_block = ""
        if msg.financial_context:
            fd = msg.financial_context
            context_block = (
                f"\n\nUser's financial profile: Income=${fd.income:,.0f}, "
                f"Filing Status={fd.filing_status}, State={fd.state}, "
                f"Expenses=${fd.expenses:,.0f}, HealthInsurance=${fd.health_insurance:,.0f}, "
                f"HomeLoan=${fd.home_loan:,.0f}, NPS=${fd.nps:,.0f}, PPF=${fd.ppf:,.0f}."
            )
        full_prompt = msg.message + context_block

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": full_prompt}],
        )
        return {"success": True, "reply": response.content[0].text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample-profiles")
async def get_sample_profiles():
    """Returns sample user profiles for demo purposes."""
    return {"profiles": FINANCIAL_RECORDS}
