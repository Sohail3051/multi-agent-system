from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA_DIR = ROOT / "data"
DB_PATH = Path(os.getenv("SUPPLY_CHAIN_DB", str(DATA_DIR / "supply_chain.db")))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
APP_NAME = "Supply Chain Crisis Intelligence"
