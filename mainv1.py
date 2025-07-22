from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image
import io
import spacy
import numpy as np
import re

app = FastAPI(title="Actomat OCR/NER Extraction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

ocr = PaddleOCR(use_angle_cls=True, lang='en')
nlp = spacy.load("xx_ent_wiki_sm")

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")

    result = ocr.predict(np.array(image))
    lines = []
    if result and isinstance(result[0], dict) and "rec_texts" in result[0]:
        lines = result[0]['rec_texts']

    # Clean up and join lines for easier regex
    full_text = "\n".join(lines)

    # Try to find values for the fields
    def find_by_keywords(keywords):
        for idx, line in enumerate(lines):
            for k in keywords:
                if k.lower() in line.lower():
                    # Return next line if it looks like a value, else return part after ":"
                    if idx+1 < len(lines) and not any(w.lower() in lines[idx+1].lower() for w in keywords):
                        return lines[idx+1].strip()
                    m = re.search(r":\s*(.+)", line)
                    if m:
                        return m.group(1).strip()
                    # Sometimes value is at the end of label line
                    parts = re.split(r"[ :]", line, maxsplit=1)
                    if len(parts) > 1:
                        return parts[-1].strip()
        return ""

    def find_by_regex(pattern, default=""):
        m = re.search(pattern, full_text)
        return m.group(1).strip() if m else default

    # Simple/naive field extraction
    structured = {
        "cnp": find_by_regex(r"\bCNP\b.*?(\d{13})"),
        "seria": find_by_regex(r"\bSERIA\b.*?([A-Z]{1,3})"),
        "nrSerie": find_by_regex(r"\bSERIA\b.*?[A-Z]{1,3}\s*([\d]{3,10})"),
        "sex": find_by_keywords(["sex", "sexe"]),
        "nume": find_by_keywords(["nume", "lastname", "nom"]),
        "prenume": find_by_keywords(["prenume", "prenom", "first name"]),
        "nationalitate": find_by_keywords(["nationalitate", "nationality", "nationalite"]),
        "loc_nastere": find_by_keywords(["loc nastere", "lieu de naissance", "place ofbirth"]),
        "adresa": find_by_keywords(["adresa", "adresse", "address"]),
        "emisa_de": find_by_keywords(["emisa de", "issued by", "delivree par"]),
        "valabilitate": find_by_keywords(["valabilitate", "validite", "validity"]),
    }

    # Fallback for nrSerie if missing
    if not structured["nrSerie"]:
        structured["nrSerie"] = find_by_regex(r"\bNR\b.*?(\d{3,10})")
    # Try to remove label words from values if present
    for k, v in structured.items():
        if v:
            v = re.sub(rf".*{k}.*?:?", "", v, flags=re.IGNORECASE).strip()
            structured[k] = v

    return structured
