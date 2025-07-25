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
from paddleocr import PaddleOCR
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

ocr = PaddleOCR(
    lang='ro',
    det_db_box_thresh=0.3,
    enable_mkldnn=True
)


# Example usage
# result = ocr.ocr(your_image_path, cls=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(f"[PIPELINE] Using model device: {model.device if hasattr(model, 'device') else 'Unknown'}")


def preprocess_image(raw_bytes, max_dim=1600, min_dim_for_upscale=900):
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

def preprocess_image_variant2(raw_bytes, max_dim=1600, min_dim_for_upscale=900):
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    w, h = image.size

    if min(w, h) < min_dim_for_upscale:
        scale = 3
        image = image.resize((w * scale, h * scale), Image.LANCZOS)
        w, h = image.size

    scale = max(w, h) / max_dim if max(w, h) > max_dim else 1
    if scale > 1:
        image = image.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.6)  # Less aggressive than variant 3
    image = image.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.EDGE_ENHANCE)     # Different sharpening than variant 3

    out_bytes = io.BytesIO()
    image.save(out_bytes, format="PNG")
    out_bytes.seek(0)
    return out_bytes


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
        results = ocr.predict(np.array(preproc_img))
        ocr_lines = []

        # results is a list of one dictionary
        if results and isinstance(results[0], dict):
            rec_texts = results[0].get("rec_texts", [])
            for text in rec_texts:
                if text.strip():
                    ocr_lines.append(text.strip())

        lines = ocr_lines
        ocr_text = " , ".join(lines)
        ocr_text = remove_5char_after_cnp(ocr_text)
        ocr_text = remove_items_with_lt(ocr_text)
        ocr_results.append(ocr_text)
        print(f"[API] OCR text (variant {idx}):")
        print(ocr_text)

    # === LLM PROMPT: Compare all OCRs and extract best fields ===
    system_prompt = (
        "Ești un asistent care extrage date structurate din text OCR provenit din buletinul romanesc. "
        "OCR-ul poate contine erori, dar trebuie să extragi informații corecte."
        "Considera ca textul poate contine artefacte cauzate de diacritice"
        "!!!Seria ('seria') este formată din exact 2 litere, iar numărul de serie ('nrSerie') este de obicei format din 6 cifre (poate fi uneori 5 sau 7). "
        "Exemplu: dacă apare textul ,,SERIA XM NR 123456”, atunci: seria = XM, nrSerie = 123456"
        "Răspunde cu un obiect JSON cu următoarele câmpuri: "
        "cnp, seria, nrSerie, sex, nume, prenume, nationalitate, loc_nastere, adresa_completa_domiciliu, emisa_de, valabilitate, strada, localitate_domiciliu, judet_nastere, judet_domiciliu, apartament, numar, etaj, scara. "
        "Dacă un câmp nu există în text, pune valoarea șir vid (''). Răspunde doar cu obiectul JSON, fără explicații. "
        "Nu o sa gasesti nici 'seria' nici 'nrSerie' in CNP, deci nu le cauta acolo. "
        "adresa_completa_domiciliu trebuie să conțină si localitatea, comuna, orasul sau municipiul, in functie de caz."
        "ATENȚIE: Câmpul 'emisa_de' trebuie să conțină numele unui oraș sau municipiu sau a unei delimitări administrative care reprezinta instituția emitentă a buletinului."
        "Județul poate să apară în text ca 'județ' sau 'jud.' urmat de numele județului. Pentru câmpurile 'judet_nastere' și 'judet_domiciliu', răspunde doar cu codul județului sau cu numele județului fără prefixe, ex: 'Maramureș' in loc de 'Jud.Maramureș'. "
        "Încearcă să identifici corect județul separat de oraș sau municipiu. "
        "Pentru adresa_completa_domiciliu NU trebuie inclus orașul sau județul. "
        "!!! ATENȚIE: Dacă NU există un nume de stradă în adresă, iar adresa conține doar localitatea și numărul (ex: 'Sat. Crasna Viseului (Com. Bistra) nr 448'), câmpul 'strada' trebuie să fie șir vid (''). NU folosi numele localității pe post de stradă! "
        "!!! Dacă găsești 'nr' sau 'număr' urmat de cifre oriunde în adresa_completa_domiciliu, pune acea valoare în 'numar' indiferent dacă strada e prezentă sau nu. "
        "Dacă vezi litere care nu există în alfabetul limbii române sau cu accent greșit, corectează-le cu litera care se potrivește. "
        "Simplifică nationalitatea într-un singur cuvânt dacă este posibil. "
        "Primești trei variante OCR, toate rezultate pe baza aceleași imagini dar cu preprocesări diferite. "
        "Compară fiecare item extras din cele trei variante și alege pentru fiecare câmp varianta care pare cea mai corectă sau completă, sau combină dacă este nevoie. "
        "Daca o valoare cautata este la fel de lunga in 2 din 3 variante, alege acea varianta"
        "Dacă un câmp nu există în niciuna, pune șir vid (''). Răspunde doar cu obiectul JSON."
        "Foloseste-ti cunostintele despre denumirile de străzi, orașe, județe și alte entități administrative din România, dar si terminatile obisnuite pentru numele de orase, comune, sate, străzi, dpdv. gramatical"
        "Foloseste-ti cunostintele despre numele de familie si prenume românești"
        "Judeca care este cea mai probabila varianta a numelui de familie in functie de localizarea geografica"
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
    response = outputs[0]['generated_text'].replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    print("[API] Raw LLM output (cleaned):")
    print(response)

    # Extract only the first valid JSON object
    json_match = re.search(r'\{.*?\}', response, flags=re.DOTALL)
    if json_match:
        json_part = json_match.group(0)
    else:
        json_part = response  # fallback to full string

    # json_start = response.find('{')
    # json_end = response.rfind('}') + 1
    # if json_start != -1 and json_end > json_start:
    #     json_part = response[json_start:json_end]
    # else:
    #     json_part = response

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
        "adresa_completa_domiciliu", "emisa_de", "valabilitate", "localitate_domiciliu",
        "judet_nastere", "judet_domiciliu"
    ]
    post_reasoning = True

    if post_reasoning:
        print("[API] Initiating post-reasoning")
        improve_prompt = (
            "Primești un obiect JSON cu câteva câmpuri cu informatii personale a unei persoane dintr-un document de identitate romanesc"
            "Aceste campuri au fost completate anterior tot de tine folsind rezultatele unui OCR, dar s-ar putea sa existe greseli asa ca ti-am dat rezultatul tau ca sa faci o ultima verificare"
            "Vei primi si rezultatele OCR pe care ai facut citirea initial"
            "Verifica daca urmatoarele criterii au fost indeplinite"
            "Criterii de indeplinit:"
            "'adresa_completa_domiciliu' trebuie sa fie o adresa completa care de obicei are un numar de casa sau de apartament"
            "Dacă campul 'adresa_completa_domiciliu' conține și numărul, dar campul 'numar' este gol, extrage numărul din adresă și completează-l în câmpul 'numar'. "
            "Campul 'localitate_nastere' trebuie sa contina doar localitatea specificata in 'adresa_completa_domiciliu'"
            "Dacă NU există un nume de stradă în adresă, iar adresa conține doar localitatea și numărul (ex: 'Sat. Crasna Viseului (Com. Bistra) nr 448'), câmpul 'strada' trebuie să fie șir vid (''). NU folosi numele localității pe post de stradă! "
            "Dacă găsești 'nr' sau 'număr' urmat de cifre oriunde în adresa_completa_domiciliu, pune acea valoare în 'numar' indiferent dacă strada e prezentă sau nu."
            "Returnează doar un nou obiect JSON corectat si nimic altceva"
            "/no-think\n"
            f"OCR intrare: {ocr_results[0]}, {ocr_results[1], {ocr_results[2]}}"
            f"JSON intrare:\n{json.dumps(data, ensure_ascii=False)}\n"
            "ATENTIE!!! Returnează doar un obiect JSON corectat si nimic altceva"
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


    # If the result includes <|im_start|> or <|im_end|>, remove it
    if isinstance(data, str):
        data = data.replace("<|im_start|>", "").replace("<|im_end|>", "")


    # Regex fallback for 'numar'
    if data.get("numar", "") == "" and data.get("adresa_completa_domiciliu", ""):
        adresa = data.get("adresa_completa_domiciliu", "")
        
        match = re.search(r'\b(?:nr\.?|număr)\s*(\d+)', adresa, flags=re.IGNORECASE)
        if match:
            data["numar"] = match.group(1)
            print(f"[API] Regex fallback: filled 'numar' with {data['numar']}")

    # If 'adresa_completa_domiciliu' contains 'bl.' or 'bloc' set 'numar' to empty
    if data.get("numar", "") != "":
        adresa = data.get("adresa_completa_domiciliu", "").lower()
        if "bl." in adresa or "bloc" in adresa:
            data["numar"] = ""

    # If seria is longer than 2 characters, truncate it
    if len(data.get("seria", "")) > 2:
        data["seria"] = data["seria"][:2]
        print(f"[API] Truncated 'seria' to {data['seria']}")

    # Ig CNP is longer than 13 characters, truncate it
    if len(data.get("cnp", "")) > 13:
        data["cnp"] = data["cnp"][:13]
        print(f"[API] Truncated 'cnp' to {data['cnp']}")

    return data
