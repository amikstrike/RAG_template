
from pypdf import PdfReader
import io
import os
from typing import Dict, Any, Optional
from common import *

# ---------- PDF & Text Extraction ----------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts).strip()


def simple_metadata_guess(raw_text: str, filename: str) -> Dict[str, Any]:
    # Heuristic: use first 5 non-empty lines for guesses
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    first = lines[0] if lines else os.path.splitext(filename)[0]
    name = first[:128]
    profession = ""
    years = None
    # naive pattern guesses
    for ln in lines[:10]:
        low = ln.lower()
        if any(k in low for k in ["developer", "engineer", "analyst", "manager", "scientist", "designer", "consultant"]):
            profession = ln[:128]
        if "years" in low and any(x in low for x in ["experience", "exp.", "exp"]):
            # try to find a number before 'years'
            import re
            m = re.search(r"(\d+(?:\.\d+)?)\s+years", low)
            if m:
                try:
                    years = float(m.group(1))
                except Exception:
                    pass
    return {"name": name, "profession": profession, "years_experience": years}

