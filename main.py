import os
import math
from io import BytesIO
from PIL import Image, ImageOps
import json
import re
import datetime
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from transformers import pipeline, AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch
from doc_mask_utils import crop_and_remove_code, rotate_90
from qwen_vl_utils import process_vision_info
from prompts import fields

model_name_or_path = "./models/Qwen2.5-VL-7B-Instruct"

# Load the vision-language model correctly
print("[INIT] Loading Qwen2.5-VL model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

print(f"[INIT] Model first parameter device: {next(model.parameters()).device}")

print(f"[INIT] CUDA available: {torch.cuda.is_available()}")
print(f"[INIT] Model device map: {model.device_map if hasattr(model, 'device_map') else 'N/A'}")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(f"[PIPELINE] Using model device: {model.device if hasattr(model, 'device') else 'Unknown'}")

def is_readable(image_bytes):
    """
    Uses Qwen2.5-VL to determine if the image appears to be a complete ID card
    from which all information can be extracted.
    """
    print ('---------', query_qwen_vl_with_image(
        image_bytes,
        prompt="Is there a surname and a valability date readable in this image? Answer only True or False"
    ))


def query_qwen_vl_with_image(image_bytes, prompt):
    """
    Query Qwen2.5-VL with an image and text prompt
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        # ✅ Resize image while keeping aspect ratio, max 1024x1024
        if max(image.width, image.height) > 1024:
            image.thumbnail((1024, 1024), Image.LANCZOS)

        
        # Prepare messages in the format expected by Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process the input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the same device as model
        inputs = inputs.to(model.device)
        
        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
        
    except Exception as e:
        print(f"[ERROR] Failed to query Qwen2.5-VL: {e}")
        return None

def query_qwen_vl_with_image(image_bytes, prompt):
    """
    Query Qwen2.5-VL with an image and text prompt
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        # ✅ Resize image while keeping aspect ratio, max 1024x1024
        if max(image.width, image.height) > 1024:
            image.thumbnail((1024, 1024), Image.LANCZOS)

        
        # Prepare messages in the format expected by Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process the input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the same device as model
        inputs = inputs.to(model.device)
        
        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
        
    except Exception as e:
        print(f"[ERROR] Failed to query Qwen2.5-VL: {e}")
        return None

# total_cores = os.cpu_count() or 4
# n_threads = max(1, math.floor(total_cores * 0.5))
# print(f"[INIT] Detected {total_cores} logical CPU cores. Using {n_threads} threads for inference.")

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
    # raw = auto_rotate(raw, jpegr_path=r".\jpegr_portable_64bit\jpegr.exe")

    original_bytes = raw
    processed_bytes = None

    for i in range(1, 4):
        print(f"[ROTATE+CROP] Attempt {i}: Cropping and checking readability...")

        result = crop_and_remove_code(original_bytes)
        if result is None:
            print(f"[ROTATE+CROP] No crop result, using full image.")
            processed_bytes = original_bytes
        else:
            cropped_img, box, dist_px, perc_right = result
            buf = io.BytesIO()
            cropped_img.save(buf, format="PNG")
            processed_bytes = buf.getvalue()

        debug_filename = f"rotated_attempt_{i}.png"
        with open(debug_filename, "wb") as f:
            f.write(processed_bytes)
        print(f"[ROTATE+CROP] Saved cropped image to {debug_filename}")

        check_result = query_qwen_vl_with_image(processed_bytes, "Is there a surname and a valability date readable in this image? Answer only True or False")
        print(f"[ROTATE+CROP] Model response: {check_result}")

        if check_result and "true" in check_result.lower():
            print(f"[ROTATE+CROP] Image is readable after {i} rotation(s). Proceeding.")
            break

        print(f"[ROTATE+CROP] Image not readable. Rotating original image 90 degrees clockwise...")
        img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        rotated = img.rotate(-90 * i, expand=True)
        buf = io.BytesIO()
        rotated.save(buf, format="PNG")
        original_bytes = buf.getvalue()
    else:
        print("[ROTATE+CROP] Tried all 4 orientations. Proceeding with the last cropped image.")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = f"cropped_{ts}.png"
    Image.open(io.BytesIO(processed_bytes)).save(debug_path)
    file_lower = file.filename.lower()
    is_image = file_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    if not is_image:
        raise HTTPException(400, "Unsupported file type. Only images are supported.")

    data = {}

    def format_context(completed_data):
        """Format the completed data as context for the next query"""
        if not completed_data:
            return ""
        
        context_items = []
        for key, value in completed_data.items():
            if value:  # Only include non-empty values
                context_items.append(f"{key}: {value}")
        
        if context_items:
            return f"\n\nFor context, here are the fields already extracted from this document:\n" + "\n".join(context_items) + "\n"
        return ""

    for key, base_prompt in fields.items():
        
        # Add context of previously completed fields
        context = format_context(data)
        full_prompt = base_prompt + context
        
        raw_response = query_qwen_vl_with_image(processed_bytes, full_prompt)
        
        if raw_response is None:
            print(f"[API] Failed to get value for {key}")
            data[key] = ""
            continue

        # Try to extract clean value
        value = raw_response.strip().strip('"').strip()
        
        # Quick heuristic to reduce hallucinated formatting
        if value.lower().startswith(key.lower() + ":"):
            value = value.split(":", 1)[-1].strip()

        data[key] = value

    adresa = data.get("adresa_completa_domiciliu", "").strip()
    loc_nastere = data.get("loc_nastere").strip()

    if loc_nastere:
        # Add context for address subfield extraction as well
        context = format_context(data)
        
        prompt = (
            f"You are given a place of birth extracted from a Romanian ID document:\n\n"
            f"\"{loc_nastere}\"\n\n"
            "Extract only the following field from this location: "
            "judet_nastere. It is a Romanian region name, not a village or city name. It is usually prefixed by 'Jud.' or 'Judet' and it can be a two-letter county code or full county name\n"
            "It's most likely one of these: AB, AR, AG, BC, BH, BN, BT, BR, BV, B, BZ, CL, CS, CJ, CT, CV, DB, DJ, GL, GR, GJ, HR, HD, IS, IL, IF, MM, MH, MS, NT, OT, PH, SJ, SM, SB, SV, TR, TM, TL, VL, VS, VN"
            "Respond with a JSON object that contains exactly this field and nothing else\n"
            "Do NOT add explanations. Do NOT include any extra fields. Respond with JSON only."
            + context
        )

        print("[API] Extracting judet nastere...")
        raw = query_qwen_vl_with_image(processed_bytes, prompt)
        print("[API] Raw address subfield response:", raw)

        json_match = re.search(r'\{.*?\}', raw or '', flags=re.DOTALL)
        if json_match:
            try:
                addr_fields = json.loads(json_match.group(0))
                for field in ["judet_nastere"]:
                    data[field] = addr_fields.get(field, "")
                    print('----', data[field])
            except Exception as e:
                print("[API] Failed to parse address JSON:", e)
                for field in ["judet_nastere"]:
                    data[field] = ""
        else:
            print("[API] Judet nastere extraction returned no JSON.")
            for field in ["judet_nastere"]:
                data[field] = ""

    if adresa:
        # Add context for address subfield extraction as well
        context = format_context(data)
        
        prompt = (
            f"You are given a full Romanian home address extracted from an ID document:\n\n"
            f"\"{adresa}\"\n\n"
            "Extract only the following fields from this address, if they are present: "
            "'strada (str.)', 'apartament (ap. or apt.)', 'numar (nr.)', 'etaj (or et.)', and 'scara (or sc.)'.\n"
            "Respond with a JSON object that contains exactly these five fields.\n"
            "For numar do not add the nr. prefix return only the number itself. If nothing found, return a dash '-'"
            "For the 'strada' field you don't have to leave out the prefix like 'Str.' or 'Bvd.', you can keep it there."
            "If a field is missing from the address, return it with a dash ('-').\n"
            "Do NOT add explanations. Do NOT include any extra fields. Respond with JSON only."
            + context
        )

        print("[API] Extracting address subfields in one call...")
        raw = query_qwen_vl_with_image(processed_bytes, prompt)
        print("[API] Raw address subfield response:", raw)

        json_match = re.search(r'\{.*?\}', raw or '', flags=re.DOTALL)
        if json_match:
            try:
                addr_fields = json.loads(json_match.group(0))
                for field in ["strada", "apartament", "numar", "etaj", "scara"]:
                    data[field] = addr_fields.get(field, "")
            except Exception as e:
                print("[API] Failed to parse address JSON:", e)
                for field in ["strada", "apartament", "numar", "etaj", "scara"]:
                    data[field] = ""
        else:
            print("[API] Address subfield extraction returned no JSON.")
            for field in ["strada", "apartament", "numar", "etaj", "scara"]:
                data[field] = ""

    return data