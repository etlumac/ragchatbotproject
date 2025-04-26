from supabase import create_client
import os
from pathlib import Path
from services.utils import safe_filename
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET_NAME", "documents")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_file(user_id: str, filename: str, file_path: Path):
    safe_name = safe_filename(filename)
    storage_path = f"{user_id}/{safe_name}"
    with open(file_path, "rb") as f:
        response = supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path,
            file=f,
            file_options={"upsert": "true"}
        )
    return response

def delete_file(user_id: str, filename: str):
    safe_name = safe_filename(filename)
    storage_path = f"{user_id}/{safe_name}"
    response = supabase.storage.from_(SUPABASE_BUCKET).remove([storage_path])
    return response

def download_file(user_id: str, filename: str, save_dir: Path = Path("downloads")) -> Path:
    safe_name = safe_filename(filename)
    storage_path = f"{user_id}/{safe_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / safe_name  # сохраняем под оригинальным именем
    res = supabase.storage.from_(SUPABASE_BUCKET).download(storage_path)
    with open(file_path, "wb") as f:
        f.write(res)
    return file_path

def list_uploaded_files(user_id: str):
    res = supabase.storage.from_(SUPABASE_BUCKET).list(path=user_id)
    files = [file["name"] for file in res if file["name"] != ".emptyFolderPlaceholder"]
    return files
