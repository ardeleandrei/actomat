# Romanian ID Multimodal Language Model Data Extractor

A FastAPI service that extracts structured fields from Romanian ID card images using a local Qwen2.5-VL-7B-Instruct model.

## Features
- Local vision-language inference (no internet needed)
- Auto-crops and rotates ID images to improve readability
- Self-correction: Since the model cannot read flipped documents and no reliable tool was found to automatically straighten them into the correct horizontal orientation, the service rotates the image up to 3 times and retries extraction until it can be read.
- Iterative field extraction with context to reduce model errors
- Special handling for Romanian-specific fields:
  - `judet_nastere` from place of birth
  - Address split into `strada`, `numar`, `apartament`, `etaj`, `scara`
- Saves debug images (`rotated_attempt_*.png`, `cropped_*.png`) for troubleshooting
- CORS enabled for frontend integration

## API
### POST /extract
**Input**: Multipart form-data with `file` (image)  
Supported formats: .png, .jpg, .jpeg, .bmp, .tiff, .webp

**Output**: JSON with extracted fields

Example:
```json
{
  "nume": "POPESCU",
  "prenume": "ION",
  "loc_nastere": "Cluj-Napoca, CJ",
  "judet_nastere": "CJ",
  "adresa_completa_domiciliu": "Str. Memorandumului nr. 12, ap. 8, et. 2, sc. A, Cluj-Napoca",
  "strada": "Str. Memorandumului",
  "numar": "12",
  "apartament": "8",
  "etaj": "2",
  "scara": "A"
}
```

## Requirements
- Python 3.10+
- GPU with CUDA recommended (works on CPU but slower)
- Model folder at `./models/Qwen2.5-VL-7B-Instruct`

Install dependencies:
```bash
pip install fastapi uvicorn pillow transformers torch accelerate paddleocr
```

Local modules required:
- `doc_mask_utils.py` (crop/remove barcode, rotate)
- `qwen_vl_utils.py` (process image input for model)
- `prompts.py` (defines extraction prompts)

## Run
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Notes
- Images larger than 1024px are resized to save VRAM
- Self-correction rotates and retries up to 3 times if image is not readable
- Debug images are saved in the working directory
