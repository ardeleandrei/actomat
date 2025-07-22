import os
import math
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import json
import re
import datetime
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import easyocr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from doc_mask_utils import crop_and_remove_code

model_name_or_path = "./models/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"[INIT] Model first parameter device: {next(model.parameters()).device}")

print(f"[INIT] CUDA available: {torch.cuda.is_available()}")
print(f"[INIT] Model device map: {model.device_map if hasattr(model, 'device_map') else 'N/A'}")

ocr = easyocr.Reader(['ro'], gpu=torch.cuda.is_available())

# Example usage
# result = ocr.ocr(your_image_path, cls=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(f"[PIPELINE] Using model device: {model.device if hasattr(model, 'device') else 'Unknown'}")


def preprocess_image(raw_bytes, max_dim=1600, min_dim_for_upscale=900):
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    w, h = image.size

    # Upscale if smallest dimension is very small (ex: scan prost, poza mică)
    if min(w, h) < min_dim_for_upscale:
        scale = 2
        image = image.resize((w*scale, h*scale), Image.LANCZOS)
        w, h = image.size

    # Downscale if it's too big for OCR pipeline (ex: poza de 6000px)
    scale = max(w, h) / max_dim if max(w, h) > max_dim else 1
    if scale > 1:
        image = image.resize((int(w/scale), int(h/scale)), Image.LANCZOS)

    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    out_bytes = io.BytesIO()
    image.save(out_bytes, format="PNG")
    out_bytes.seek(0)
    return out_bytes

def preprocess_image_variant2(raw_bytes, max_dim=1600, min_dim_for_upscale=900):
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    w, h = image.size

    if min(w, h) < min_dim_for_upscale:
        scale = 3
        image = image.resize((w*scale, h*scale), Image.LANCZOS)
        w, h = image.size

    scale = max(w, h) / max_dim if max(w, h) > max_dim else 1
    if scale > 1:
        image = image.resize((int(w/scale), int(h/scale)), Image.LANCZOS)

    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    out_bytes = io.BytesIO()
    image.save(out_bytes, format="PNG")
    out_bytes.seek(0)
    return out_bytes

import re

def clean_ocr_output(ocr_text):
    # Fix common character distortions
    ocr_text = ocr_text.replace("ROMAITJTF", "ROMÂNIA")  # Example of custom replacement
    ocr_text = ocr_text.replace("ROMANIJTE", "ROMÂNIA")
    ocr_text = ocr_text.replace("delDelivree", "Delivered by")
    ocr_text = ocr_text.replace("parllssued", "Issued")
    ocr_text = ocr_text.replace("CNP", "CNP")  # Ensure CNP remains the same
    
    # Remove unwanted characters or symbols
    ocr_text = re.sub(r'[^a-zA-Z0-9ăîșțâĂÎȘȚÂ\.\,\(\)\-\s]', '', ocr_text)  # Remove non-alphanumeric chars except Romanian letters and common punctuation
    
    # Fix spacing issues (remove extra spaces or replace with a single space)
    ocr_text = re.sub(r'\s+', ' ', ocr_text)
    
    # Replace distorted letters (e.g., OCR might detect 'I' as 'l' or 'J' as 'I')
    ocr_text = ocr_text.replace("I", "J")  # Fix wrongly detected 'I' as 'J' in some cases
    
    # Handle missing accents in Romanian words
    ocr_text = ocr_text.replace("Română", "Română")
    ocr_text = ocr_text.replace("Maramureș", "Maramureș")
    ocr_text = ocr_text.replace("Bistra", "Bistra")
    
    # Further cleanup (remove any remaining symbols or fix specific distortions)
    ocr_text = ocr_text.replace("Sat. ", "Sat")  # Remove dots from 'Sat.'
    ocr_text = re.sub(r'(\d)\s*(\d)', r'\1\2', ocr_text)  # Merge numbers with spaces (e.g., 448 -> 448)
    
    # Fix common incorrect abbreviations (e.g., 'Jud' should be 'Judet')
    ocr_text = ocr_text.replace("Jud.", "Județ")
    
    return ocr_text.strip()

def preprocess_image_variant3(raw_bytes, max_dim=1600, min_dim_for_upscale=900):
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    w, h = image.size

    if min(w, h) < min_dim_for_upscale:
        scale = 4
        image = image.resize((w*scale, h*scale), Image.LANCZOS)
        w, h = image.size

    scale = max(w, h) / max_dim if max(w, h) > max_dim else 1
    if scale > 1:
        image = image.resize((int(w/scale), int(h/scale)), Image.LANCZOS)

    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(2.2)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.DETAIL)
    image = image.filter(ImageFilter.MedianFilter(size=3))

    out_bytes = io.BytesIO()
    image.save(out_bytes, format="PNG")
    out_bytes.seek(0)
    return out_bytes

# === Remove any 5-char item that comes right after 'CNP' in the OCR ===
def remove_5char_after_cnp(ocr_text):
    items = [x.strip() for x in ocr_text.split(',')]
    cleaned_items = []
    skip_next = False
    for i, item in enumerate(items):
        if skip_next:
            # skip this item and reset the flag
            skip_next = False
            continue
        if item.strip().upper() == "CNP" and i + 1 < len(items):
            next_item = items[i + 1].strip()
            if len(next_item) == 5:
                skip_next = True  # skip next
        cleaned_items.append(item)
    return ' , '.join(cleaned_items)

def remove_items_with_lt(ocr_text):
    items = [x.strip() for x in ocr_text.split(',')]
    items = [item for item in items if '<' not in item]
    return ' , '.join(items)

total_cores = os.cpu_count() or 4
n_threads = max(1, math.floor(total_cores * 0.5))
print(f"[INIT] Detected {total_cores} logical CPU cores. Using {n_threads} threads for inference.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# converter = DocumentConverter()

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    print("[API] File received:", file.filename)
    raw = await file.read()

    result = crop_and_remove_code(raw)
    if result is None:
        # fallback to original if no person detected
        processed_bytes = raw
    else:
        cropped_img, box, dist_px, perc_right = result
        buf = io.BytesIO()
        cropped_img.save(buf, format="PNG")
        processed_bytes = buf.getvalue()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = f"cropped_{ts}.png"
    Image.open(io.BytesIO(processed_bytes)).save(debug_path)
    print(f"[DEBUG] Cropped document image saved to {debug_path}")

    file_lower = file.filename.lower()
    is_image = file_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    if not is_image:
        raise HTTPException(400, "Unsupported file type. Only images are supported.")

    # === Preprocess & OCR: run all 3 variants ===
    preprocessed1 = preprocess_image(processed_bytes, max_dim=1600)
    preprocessed2 = preprocess_image_variant2(processed_bytes, max_dim=1600)
    preprocessed3 = preprocess_image_variant3(processed_bytes, max_dim=1600)
    import numpy as np
    ocr_results = []
    for idx, preproc_img_bytes in enumerate([preprocessed1, preprocessed2, preprocessed3], start=1):
        preproc_img = Image.open(preproc_img_bytes).convert("RGB")
        results = ocr.readtext(np.array(preproc_img))
        lines = [item[1] for item in results if item[1].strip()]
        ocr_text = " , ".join(lines)
        ocr_text = remove_5char_after_cnp(ocr_text)
        ocr_text = remove_items_with_lt(ocr_text)
        ocr_results.append(ocr_text)
        print(f"[API] OCR text (variant {idx}):")
        print(ocr_text)

    # === LLM PROMPT: Compare all OCRs and extract best fields ===
    system_prompt = (
        "Ești un asistent care extrage date structurate din text OCR provenit din buletinul romanesc. "
        "Asigură-te că folosești doar diacriticele corecte din limba română pentru literele cu accent ('ă', 'î', 'ș', 'ț', 'â', 'Ă', 'Î', 'Ș', 'Ț', 'Â') "
        "!!!Seria ('seria') este formată din exact 2 litere, iar numărul de serie ('nrSerie') este de obicei format din 6 cifre (poate fi uneori 5 sau 7). "
        "Răspunde cu un obiect JSON cu următoarele câmpuri: "
        "cnp, seria, nrSerie, sex, nume, prenume, nationalitate, loc_nastere, adresa_completa_domiciliu, emisa_de, valabilitate, strada, loc_domiciliu, judet_nastere, judet_domiciliu, apartament, numar, etaj, scara. "
        "Dacă un câmp nu există în text, pune valoarea șir vid (''). Răspunde doar cu obiectul JSON, fără explicații. "
        "Județul poate să apară în text ca 'județ' sau 'jud.' urmat de numele județului. Pentru câmpurile 'judet_nastere' și 'judet_domiciliu', răspunde doar cu codul județului sau cu numele județului fără prefixe, ex: 'Maramureș' in loc de 'Jud.Maramureș'. "
        "Încearcă să identifici corect județul separat de oraș sau municipiu. "
        "Pentru adresa_completa_domiciliu NU trebuie inclus orașul sau județul. "
        "!!! ATENȚIE: Dacă NU există un nume de stradă în adresă, iar adresa conține doar localitatea și numărul (ex: 'Sat. Crasna Viseului (Com. Bistra) nr 448'), câmpul 'strada' trebuie să fie șir vid (''). NU folosi numele localității pe post de stradă! "
        "!!! Dacă găsești 'nr' sau 'număr' urmat de cifre oriunde în adresa_completa_domiciliu, pune acea valoare în 'numar' indiferent dacă strada e prezentă sau nu. "
        "Dacă vezi litere care nu există în alfabetul limbii române sau cu accent greșit, corectează-le cu litera care se potrivește. "
        "Ignoră orice linie care conține multe caractere '<' (MRZ), acestea nu sunt date valide. "
        "Simplifică nationalitatea într-un singur cuvânt dacă este posibil. "
        "Primești trei variante OCR, toate rezultate pe baza aceleiași imagini dar cu preprocesări diferite. "
        "Compară fiecare item extras din cele trei variante și alege pentru fiecare câmp varianta care pare cea mai corectă sau completă, sau combină dacă este nevoie. "
        "Dacă un câmp nu există în niciuna, pune șir vid (''). Răspunde doar cu obiectul JSON."
        "Răspunde doar cu obiectul JSON. NU adăuga explicații, nu folosi Markdown, nu adăuga nimic altceva decât obiectul JSON."
        "Return ONLY the JSON object. Do NOT add explanations, do NOT use Markdown, do NOT add anything else except the JSON object."

    )
    user_prompt = (
        f"OCR varianta 1:\n{ocr_results[0]}\n\n"
        f"OCR varianta 2:\n{ocr_results[1]}\n\n"
        f"OCR varianta 3:\n{ocr_results[2]}"
    )
    prompt = (
        "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
        "<|im_start|>user\n" + user_prompt + "<|im_end|>\n"
    )

    print("[API] Sending prompt to LLM (compare OCR variants)...")
    print(f"[API] Running LLM pipeline on device: {model.device if hasattr(model, 'device') else 'Unknown'}")

    outputs = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False,
    )
    response = outputs[0]['generated_text'].strip()

    print("[API] Raw LLM output:")
    print(response)

    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    if json_start != -1 and json_end > json_start:
        json_part = response[json_start:json_end]
    else:
        json_part = response

    try:
        data = json.loads(json_part)
        print("[API] Successfully parsed JSON.")
    except Exception as e:
        print("[API] Failed to parse JSON:", e)
        data = {"raw_output": response}
        return data

    # === LLM PROMPT 2 (if needed) ===
    required_fields = [
        "cnp", "seria", "nrSerie", "sex", "nume", "prenume", "nationalitate", "loc_nastere",
        "adresa_completa_domiciliu", "emisa_de", "valabilitate", "loc_domiciliu",
        "judet_nastere", "judet_domiciliu"
    ]
    needs_retry = isinstance(data, dict) and any(
        data.get(field, "") == "" for field in required_fields
    )

    if needs_retry:
        print("[API] Empty fields detected. Running second LLM trial to complete missing fields using only JSON.")
        improve_prompt = (
            "Ai primit deja un obiect JSON cu câteva câmpuri goale (''). "
            "Încearcă să completezi câmpurile lipsă folosind doar datele deja extrase din celelalte câmpuri ale JSON-ului. "
            "De exemplu, dacă 'adresa_completa_domiciliu' conține și numărul, dar 'numar' este gol, extrage numărul din adresă și completează-l în câmpul 'numar'. "
            "!!! ATENȚIE: Dacă NU există un nume de stradă în adresă, iar adresa conține doar localitatea și numărul (ex: 'Sat. Crasna Viseului (Com. Bistra) nr 448'), câmpul 'strada' trebuie să fie șir vid (''). NU folosi numele localității pe post de stradă! "
            "!!! Dacă găsești 'nr' sau 'număr' urmat de cifre oriunde în adresa_completa_domiciliu, pune acea valoare în 'numar' indiferent dacă strada e prezentă sau nu.\n"
            "Returnează un nou obiect JSON cu aceleași câmpuri, completând tot ce se poate, fără explicații."
            "Pentru câmpurile 'judet_nastere' și 'judet_domiciliu', răspunde doar cu codul județului sau cu numele județului fără prefixe, ex: 'Maramureș' in loc de 'Jud.Maramureș'.\n"
            "/no-think\n"
            f"JSON parțial:\n{json.dumps(data, ensure_ascii=False)}"
        )
        improve_full_prompt = (
            "<|im_start|>system\n" + improve_prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        outputs2 = pipe(
            improve_full_prompt,
            max_new_tokens=512,
            return_full_text=False,
        )
        response2 = outputs2[0]['generated_text'].strip()
        stop_str = "<|im_end|>"
        stop_idx = response2.find(stop_str)
        if stop_idx != -1:
            response2 = response2[:stop_idx].strip()
        print("[API] Raw LLM output (second trial):")
        print(response2)

        json_start2 = response2.find('{')
        json_end2 = response2.rfind('}') + 1
        if json_start2 != -1 and json_end2 > json_start2:
            json_part2 = response2[json_start2:json_end2]
        else:
            json_part2 = response2

        try:
            data2 = json.loads(json_part2)
            print("[API] Successfully parsed JSON (second trial).")
            data = data2
        except Exception as e:
            print("[API] Failed to parse JSON (second trial):", e)
            # rămâi pe primul
            pass



    # Regex fallback for 'numar'
    if data.get("numar", "") == "" and data.get("adresa_completa_domiciliu", ""):
        adresa = data.get("adresa_completa_domiciliu", "")
        import re
        match = re.search(r'\b(?:nr\.?|număr)\s*(\d+)', adresa, flags=re.IGNORECASE)
        if match:
            data["numar"] = match.group(1)
            print(f"[API] Regex fallback: filled 'numar' with {data['numar']}")

    return data
