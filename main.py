import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Initialize FastAPI App ---
app = FastAPI(title="Ticket Classification & Priority Prediction API")

# --- 1. Load Text Models & Tokenizers ---
# Pointing to your local directory containing the fine-tuned DistilBERT config and weights
# --- 1. Load Text Models & Tokenizers ---
TEXT_MODEL_DIR = Path("D:\Customer Support\DistilBert\models\distilbert_ticket_type").resolve()

# Diagnostic verification check to make sure the folder actually holds the model configuration files
if not TEXT_MODEL_DIR.joinpath("config.json").exists():
    raise FileNotFoundError(
        f"Could not find config.json at {TEXT_MODEL_DIR}.\n"
        "Please check your folder structure and ensure your files are placed here."
    )

# Load by passing the Path object directly (bypasses Windows string repo-id parsing bugs)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR, num_labels=5) # 5 labels for your 5 categories
text_model.eval()
print("✅ NLP Model successfully loaded into memory!")

# --- 2. Tabular Priority Predictor Fallback ---
# Since you don't have the joblib file, this substitutes the priority pipeline logic via python rules.
def predict_priority_fallback(category: str, user_tier: str, days_open: int) -> str:
    tier_scores = {"Male": 1, "Female": 1, "Other": 1} # Adjusted to map incoming gender values safely
    tier_score = tier_scores.get(user_tier, 0)
    
    cat_lower = category.lower()
    # High-urgency product hardware groups trigger priority boosts
    if "iphone" in cat_lower or "playstation" in cat_lower or "xbox" in cat_lower:
        cat_score = 3
    elif "macbook" in cat_lower or "dell" in cat_lower:
        cat_score = 2
    else:
        cat_score = 0
        
    total_score = tier_score + cat_score + (days_open * 0.5)
    
    if total_score >= 5:
        return "Critical"
    elif total_score >= 3:
        return "High"
    elif total_score >= 1.5:
        return "Medium"
    else:
        return "Low"


# --- 3. Define Request/Response Schemas ---
class TicketRequest(BaseModel):
    text: str
    category: str
    user_tier: str
    days_open: int


class PredictionResponse(BaseModel):
    ticket_category: str
    predicted_priority: str
    category_confidence: float


# --- 4. Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TicketRequest):
    try:
        # Step A: DistilBERT Text Inference
        inputs = tokenizer(payload.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = text_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_class_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][pred_class_idx].item()
        
        # Mapping index to text category labels from config.json
        category_mapping = text_model.config.id2label
        predicted_category = category_mapping.get(pred_class_idx, f"Class_{pred_class_idx}")

        # Step B: Tabular Priority Inference (Fallback Function)
        priority_prediction = predict_priority_fallback(
            category=payload.category, 
            user_tier=payload.user_tier, 
            days_open=payload.days_open
        )

        return PredictionResponse(
            ticket_category=predicted_category,
            predicted_priority=str(priority_prediction),
            category_confidence=round(confidence, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))