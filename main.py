from datetime import date
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

model = joblib.load("category_classifier_model.joblib")
clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

model_name = type(clf).__name__
features = ["Amount", "Month", "Quarter", "DayOfWeek", "IsWeekend", "Type"]
classes = sorted(clf.classes_.tolist())

test_accuracy = 0.260
test_macro_f1 = 0.257
baseline_accuracy = 0.110

# Budget thresholds
BUDGETS = {
    "Entertainment": 2910.39,
    "Food & Drink": 3189.87,
    "Health & Fitness": 2852.02,
    "Rent": 3301.54,
    "Shopping": 2937.62,
    "Travel": 3452.74,
    "Utilities": 2884.24,
}

# App & schemas
app = FastAPI(
    title="Personal Finance Category & Budget API",
    description=(
        "Predicts the spending category of a transaction and checks it "
        "against historical monthly budgets, for use by the n8n agentic "
        "workflow."
    ),
)

class TransactionIn(BaseModel):
    amount: float = Field(..., gt=0, examples=[125.50])
    type: Literal["Income", "Expense"] = Field(..., examples=["Expense"])
    transaction_date: Optional[date] = Field(
        default=None, description="Defaults to today if omitted", examples=["2026-07-19"]
    )


class PredictionOut(BaseModel):
    predicted_category: str
    confidence: float
    probabilities: dict[str, float]


class BudgetCheckIn(BaseModel):
    category: str
    month_to_date_spent: float = Field(..., ge=0, description="Sum already spent in this category this month, including the new transaction")


class BudgetCheckOut(BaseModel):
    category: str
    monthly_budget: Optional[float]
    month_to_date_spent: float
    remaining: Optional[float]
    over_budget: bool
    pct_of_budget_used: Optional[float]


def _row_from_transaction(tx: TransactionIn) -> pd.DataFrame:
    d = tx.transaction_date or date.today()
    return pd.DataFrame(
        [{
            "Amount": tx.amount,
            "Month": d.month,
            "Quarter": (d.month - 1) // 3 + 1,
            "DayOfWeek": d.weekday(),
            "IsWeekend": int(d.weekday() >= 5),
            "Type": tx.type,
        }]
    )

# Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "model": model_name}


@app.get("/model_info")
def model_info():
    return {
        "model_name": model_name,
        "features": features,
        "classes": classes,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "baseline_accuracy": baseline_accuracy,
    }


@app.get("/budgets")
def get_budgets():
    return BUDGETS


@app.post("/predict", response_model=PredictionOut)
def predict(tx: TransactionIn):
    row = _row_from_transaction(tx)
    proba = model.predict_proba(row)[0]
    proba_map = {str(c): float(p) for c, p in zip(clf.classes_, proba)}
    best_idx = proba.argmax()
    return PredictionOut(
        predicted_category=str(clf.classes_[best_idx]),
        confidence=float(proba[best_idx]),
        probabilities=proba_map,
    )


@app.post("/check_budget", response_model=BudgetCheckOut)
def check_budget(payload: BudgetCheckIn):
    budget = BUDGETS.get(payload.category)
    if budget is None:
        return BudgetCheckOut(
            category=payload.category,
            monthly_budget=None,
            month_to_date_spent=payload.month_to_date_spent,
            remaining=None,
            over_budget=False,
            pct_of_budget_used=None,
        )
    remaining = round(budget - payload.month_to_date_spent, 2)
    return BudgetCheckOut(
        category=payload.category,
        monthly_budget=budget,
        month_to_date_spent=payload.month_to_date_spent,
        remaining=remaining,
        over_budget=payload.month_to_date_spent > budget,
        pct_of_budget_used=round(payload.month_to_date_spent / budget * 100, 1),
    )


@app.post("/predict_and_check_budget")
def predict_and_check_budget(tx: TransactionIn, month_to_date_spent_excl_this_tx: float = 0.0):
    pred = predict(tx)
    total_spent = month_to_date_spent_excl_this_tx + tx.amount
    budget_check = check_budget(BudgetCheckIn(category=pred.predicted_category, month_to_date_spent=total_spent))
    return {"prediction": pred, "budget_check": budget_check}