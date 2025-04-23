import re
import unicodedata
from slugify import slugify
from pathlib import Path

def safe_filename(filename: str) -> str:
    name, ext = Path(filename).stem, Path(filename).suffix
    return f"{slugify(name)}{ext}"