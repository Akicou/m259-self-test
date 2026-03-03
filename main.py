import os
import json
import random
from typing import Optional
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="M259 Study App")
templates = Jinja2Templates(directory="templates")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
)

MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")

COURSE_CONTENT = """
M259 - ICT-Lösungen mit Machine Learning

TAG 1 - Grundlagen:
- Unterschied AI vs ML
- ML Kategorien: Supervised, Unsupervised, Reinforcement Learning
- Datentypen (Personendaten, schützenswerte Daten, Profiling) gemäss Schweizer DSG
- Algorithmic Bias: Historic Bias, Representation Bias, Measurement Bias
- Fairness-Definitionen: Demographic Parity, Equalized Odds, Individual Fairness
- Bias-Reduzierung: Pre-Processing, In-Processing, Post-Processing
- Datenschutzmassnahmen: Anonymisierung, Zugriffsregeln, sichere Speicherung
- Google Colab, Pandas DataFrames, Jupyter Notebooks

TAG 2 - Projektmanagement & EDA:
- 7 Phasen ML-Projekt: Problemdefinition, Datenakquise, Datenvorbereitung, Modellauswahl, Modellbewertung, Deployment, Wartung
- ML-Aufgaben: Klassifikation, Regression, Clustering
- Methodologien: CRISP-DM, TDSP, MLOps
- EDA: Mean, Median, Standardabweichung, Quartile, Korrelationsmatrizen
- Visualisierungen: Histogramme, Boxplots, Scatterplots
- Datenbereinigung: NULL Values, Duplikate, Ausreisser (IQR-Methode)
- Entscheidungsbäume, Lineare Regression
- Train-Test-Validation Split

TAG 3 - LLMs:
- Tokenisierung, Transformer-Architektur, Training vs Inference
- LLM-Landschaft: GPT, Claude, Gemini, LLaMA
- Prompt Engineering: Few-Shot, Chain-of-Thought, Self-Consistency
- RAG (Retrieval Augmented Generation)
- Vector Databases, Similarity Search
- Function Calling/Tool Use
- N8N Workflow-Automatisierung
- Token-System, Context-Limits
- LLM-Limitationen: Halluzinationen, Wissensgrenzen, Bias

TripBot: Reiseempfehlungssystem mit LLM-Integration, RAG, API-Anbindungen
"""

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


# Models
class FlashcardRequest(BaseModel):
    topic: str
    count: int = 5
    difficulty: str = "medium"

class ExamRequest(BaseModel):
    topic: str
    question_count: int = 5
    difficulty: str = "medium"
    question_types: list[str] = ["multiple_choice", "open"]

class AnswerRequest(BaseModel):
    question: str
    user_answer: str
    context: str = ""


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 1. Flashcards
@app.get("/flashcards", response_class=HTMLResponse)
async def flashcards_page(request: Request):
    return templates.TemplateResponse("flashcards.html", {"request": request})


@app.post("/api/flashcards")
async def generate_flashcards(
    topic: str = Form(...),
    count: int = Form(5),
    difficulty: str = Form("medium")
):
    system_prompt = """Du bist ein erfahrener Lehrer für Machine Learning und KI.
Erstelle Lernkarten im JSON-Format als Array:
[{"front": "Frage/Begriff", "back": "Antwort/Erklärung"}]"""
    
    user_prompt = f"""Erstelle {count} Lernkarten zum Thema "{topic}" für die M259 Prüfung.
Schwierigkeit: {difficulty}

Kursinhalt als Referenz:
{COURSE_CONTENT}

Wichtig: Gib NUR das JSON-Array zurück, ohne Markdown-Codeblöcke."""

    result = call_llm(system_prompt, user_prompt)
    try:
        json_str = result.replace("```json", "").replace("```", "").strip()
        flashcards = json.loads(json_str)
    except:
        flashcards = [{"front": "Error", "back": result}]
    
    return {"flashcards": flashcards}


# 2. Exam Generator
@app.get("/exam", response_class=HTMLResponse)
async def exam_page(request: Request):
    return templates.TemplateResponse("exam.html", {"request": request})


@app.post("/api/exam")
async def generate_exam(
    topic: str = Form(...),
    question_count: int = Form(5),
    difficulty: str = Form("medium"),
    question_types: str = Form("multiple_choice,open")
):
    types_list = [t.strip() for t in question_types.split(",")]
    
    system_prompt = """Du bist ein PrüfungsErsteller für M259.
Erstelle eine Prüfung im JSON-Format:
{
  "title": "Prüfungstitel",
  "questions": [
    {"id": 1, "type": "multiple_choice", "question": "...", "options": ["A", "B", "C", "D"], "correct": "A", "points": 2},
    {"id": 2, "type": "open", "question": "...", "sample_answer": "...", "points": 4}
  ]
}

WICHTIG bei multiple_choice: Es gibt IMMER GENAU EINE richtige Antwort (Single-Choice, NICHT Multi-Select!).
Das Feld "correct" enthält genau EINEN Buchstaben: "A", "B", "C" oder "D". NIEMALS mehrere Antworten!"""
    
    user_prompt = f"""Erstelle eine Prüfung zum Thema "{topic}" für M259.
Anzahl Fragen: {question_count}
Schwierigkeit: {difficulty}
Fragentypen: {types_list}

Kursinhalt:
{COURSE_CONTENT}

Gib NUR das JSON zurück."""

    result = call_llm(system_prompt, user_prompt, temperature=0.5)
    try:
        json_str = result.replace("```json", "").replace("```", "").strip()
        exam = json.loads(json_str)
    except:
        exam = {"title": "Error", "questions": []}
    
    return {"exam": exam}


# 3. Answer Judge
@app.get("/answer", response_class=HTMLResponse)
async def answer_page(request: Request):
    return templates.TemplateResponse("answer.html", {"request": request})


@app.post("/api/judge")
async def judge_answer(
    question: str = Form(...),
    user_answer: str = Form(...),
    context: str = Form("")
):
    system_prompt = """Du bist ein strenger aber fairer Prüfer für M259.
Bewerte die Antwort und gib Feedback im JSON-Format:
{
  "score": 0-100,
  "correctness": "correct/partially_correct/incorrect",
  "feedback": "Konstruktives Feedback",
  "improvement_tips": ["Tipp1", "Tipp2"],
  "model_answer": "Musterantwort"
}"""
    
    full_context = f"Kontext: {context}\n\n" if context else ""
    user_prompt = f"""{full_context}Frage: {question}

Antwort des Studierenden: {user_answer}

Bewerte diese Antwort basierend auf dem M259 Kursinhalt.
Gib NUR das JSON zurück."""

    result = call_llm(system_prompt, user_prompt, temperature=0.3)
    try:
        json_str = result.replace("```json", "").replace("```", "").strip()
        judgment = json.loads(json_str)
    except:
        judgment = {"score": 0, "feedback": result}
    
    return {"judgment": judgment}


@app.post("/api/question")
async def generate_question(topic: str = Form(...)):
    system_prompt = """Du bist ein Prüfer für M259. Generiere eine offene Frage.
JSON-Format: {"question": "...", "context": "...", "key_points": ["Punkt1", "Punkt2"]}"""
    
    user_prompt = f"""Generiere eine Prüfungsfrage zum Thema "{topic}" für M259.
Kursinhalt: {COURSE_CONTENT}
Gib NUR JSON zurück."""

    result = call_llm(system_prompt, user_prompt)
    try:
        json_str = result.replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except:
        return {"question": result, "context": "", "key_points": []}


# 4. Quiz Mode
@app.get("/quiz", response_class=HTMLResponse)
async def quiz_page(request: Request):
    return templates.TemplateResponse("quiz.html", {"request": request})


@app.post("/api/quiz")
async def generate_quiz(
    topic: str = Form(...),
    count: int = Form(10)
):
    system_prompt = """Du erstellst Quizfragen für M259.
JSON-Format: {"questions": [{"question": "...", "options": ["A","B","C","D"], "correct": 0, "explanation": "..."}]}

WICHTIG: "correct" ist der Index der EINZIGEN richtigen Antwort (0, 1, 2 oder 3).
Es ist ein Single-Choice Quiz - es gibt IMMER GENAU EINE richtige Antwort, NIEMALS mehrere!
"correct" ist eine einzelne Zahl, kein Array."""
    
    user_prompt = f"""Erstelle {count} Multiple-Choice Quizfragen zum Thema "{topic}".
Kursinhalt: {COURSE_CONTENT}
Gib NUR JSON zurück."""

    result = call_llm(system_prompt, user_prompt)
    try:
        json_str = result.replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except:
        return {"questions": []}


# 5. Summary Generator
@app.get("/summary", response_class=HTMLResponse)
async def summary_page(request: Request):
    return templates.TemplateResponse("summary.html", {"request": request})


@app.post("/api/summary")
async def generate_summary(topic: str = Form(...), style: str = Form("detailed")):
    system_prompt = """Du erstellst Zusammenfassungen für M259.
Strukturiere die Antwort mit Überschriften und Bullet Points."""
    
    style_instructions = {
        "detailed": "Ausführliche Zusammenfassung mit Beispielen",
        "bullet": "Kompakte Bullet-Point Liste",
        "exam": "Prüfungsrelevante Kernpunkte"
    }
    
    user_prompt = f"""Erstelle eine {style_instructions.get(style, 'detailed')} Zusammenfassung zum Thema "{topic}".
Kursinhalt: {COURSE_CONTENT}"""

    return {"summary": call_llm(system_prompt, user_prompt)}


# 6. Study Notes
@app.get("/notes", response_class=HTMLResponse)
async def notes_page(request: Request):
    return templates.TemplateResponse("notes.html", {"request": request})


@app.post("/api/notes")
async def generate_notes(topic: str = Form(...)):
    system_prompt = """Du erstellst strukturierte Lernnotizen für M259.
Format mit Markdown:
# Hauptthema
## Unterthema
- Punkte
### Wichtige Konzepte
### Beispiele"""
    
    user_prompt = f"""Erstelle detaillierte Lernnotizen zum Thema "{topic}" für die M259 Prüfung.
Kursinhalt: {COURSE_CONTENT}"""

    return {"notes": call_llm(system_prompt, user_prompt)}


# 7. Progress Tracker
@app.get("/progress", response_class=HTMLResponse)
async def progress_page(request: Request):
    return templates.TemplateResponse("progress.html", {"request": request})


@app.post("/api/rate-open-answer")
async def rate_open_answer(
    question: str = Form(...),
    user_answer: str = Form(...),
    max_points: int = Form(4)
):
    system_prompt = """Du bist ein strenger Prüfer für M259. Bewerte offene Antworten.
Gib das Ergebnis IMMER im JSON-Format zurück:
{
  "points": 0-max_points (als Zahl),
  "max_points": max_points,
  "percentage": 0-100,
  "feedback": "Kurzes Feedback was gut war und was fehlte",
  "key_points_covered": ["Punkt1", "Punkt2"],
  "key_points_missed": ["Punkt3"]
}"""
    
    user_prompt = f"""Bewerte diese Antwort für die M259 Prüfung.

Frage: {question}

Antwort des Studierenden: {user_answer}

Maximale Punkte: {max_points}

Bewerte basierend auf dem Kursinhalt:
{COURSE_CONTENT}

Gib NUR das JSON zurück."""

    result = call_llm(system_prompt, user_prompt, temperature=0.3)
    try:
        json_str = result.replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except:
        return {"points": 0, "max_points": max_points, "percentage": 0, "feedback": result, "key_points_covered": [], "key_points_missed": []}


@app.post("/api/analyze-progress")
async def analyze_progress(topics_known: str = Form(...), topics_struggling: str = Form(...)):
    system_prompt = """Du analysierst den Lernfortschritt für M259.
JSON-Format: {
  "overall_score": 0-100,
  "strengths": ["Stärke1"],
  "weaknesses": ["Schwäche1"],
  "recommendations": ["Empfehlung1"],
  "priority_topics": ["Priorität1"],
  "study_plan": {"day1": "...", "day2": "..."}
}"""
    
    user_prompt = f"""Analysiere den Lernfortschritt:
Beherrschte Themen: {topics_known}
Schwierige Themen: {topics_struggling}

Kursinhalt: {COURSE_CONTENT}
Gib NUR JSON zurück."""

    result = call_llm(system_prompt, user_prompt, temperature=0.5)
    try:
        json_str = result.replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except:
        return {"overall_score": 50, "recommendations": [result]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
