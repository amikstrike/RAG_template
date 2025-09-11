from datetime import datetime
import os

# ---------- Paths & Constants ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(APP_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
TEXT_DIR = os.path.join(DATA_DIR, "text")
DB_PATH = os.path.join(DATA_DIR, "candidates.db")
COLLECTION_NAME = "cvs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {msg}")
    
    