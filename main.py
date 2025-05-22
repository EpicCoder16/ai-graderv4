from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel
import shutil, os, logging
import docx, pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer, util
import db
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize FastAPI
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://epiccoder16.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Handle preflight requests
@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str = ""):
    return JSONResponse(status_code=200, content={})

# Load model once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')
answer_key_text = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper functions
def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return '\n'.join(para.text for para in doc.paragraphs)

def extract_text_from_pdf(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        return ''.join(page.extract_text() for page in pdf.pages if page.extract_text())

def compare_with_answer_key(extracted_text: str, answer_key: str, threshold: float = 0.6):
    # Compute overall similarity between full responses
    extracted_embedding = model.encode(extracted_text, convert_to_tensor=True)
    answer_embedding = model.encode(answer_key, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(extracted_embedding, answer_embedding).item()

    # Tokenize both into sentences
    extracted_sents = sent_tokenize(extracted_text)
    key_sents = sent_tokenize(answer_key)

    matched = []
    missing = []
    max_similarities = []

    if not key_sents or not extracted_sents:
        return {
            "similarity_score": float(similarity_score),
            "overall_semantic_coverage": 0.0,
            "message": "No sentences to compare.",
            "matched_points": [],
            "missing_points": key_sents
        }

    # Encode sentence embeddings
    key_embeddings = model.encode(key_sents, convert_to_tensor=True)
    extracted_embeddings = model.encode(extracted_sents, convert_to_tensor=True)

    for i, key_emb in enumerate(key_embeddings):
        similarities = util.pytorch_cos_sim(key_emb, extracted_embeddings)[0].cpu().numpy()
        max_sim = float(np.max(similarities))  # best match score for this key sentence
        max_similarities.append(max_sim)

        if max_sim >= threshold:
            matched.append(key_sents[i])
        else:
            missing.append(key_sents[i])

    return {
        "similarity_score": float(similarity_score),
        "overall_semantic_coverage": float(np.mean(max_similarities)),
        "message": "Comparison complete with semantic point analysis.",
        "matched_points": matched,
        "missing_points": missing
    }

# Upload answer key
@app.post("/api/upload_answer_key/")
async def upload_answer_key(file: UploadFile = File(...)):
    global answer_key_text
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith(".docx"):
        answer_key_text = extract_text_from_docx(file_path)
    elif file.filename.endswith(".pdf"):
        answer_key_text = extract_text_from_pdf(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a .docx or .pdf.")

    return {"filename": file.filename, "message": "Answer key uploaded successfully."}

# Upload student file
@app.post("/api/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    if not answer_key_text:
        raise HTTPException(status_code=400, detail="Answer key not uploaded yet.")

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith(".docx"):
        extracted_text = extract_text_from_docx(file_path)
    elif file.filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    result = compare_with_answer_key(extracted_text, answer_key_text)
    db.store_comparison(user_id, file.filename, result["similarity_score"])

    return {
        "filename": file.filename,
        "extracted_text": extracted_text,
        "comparison_result": result
    }

# Get comparisons
@app.get("/api/comparisons/{user_id}")
def get_user_comparisons(user_id: str):
    return db.get_user_comparisons(user_id)

# User model
class User(BaseModel):
    username: str
    password: str

# Register user
@app.post("/api/register")
def register(user: User):
    if not db.register_user(user.username, user.password):
        raise HTTPException(status_code=400, detail="Username already exists.")
    return {"message": "User registered successfully."}

# Login user
@app.post("/api/login")
def login(user: User):
    user_id = db.login_user(user.username, user.password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {"message": "Login successful", "user_id": user_id}
