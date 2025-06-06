import bcrypt
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(username, email, password):
    hashed_pw = hash_password(password)
    result = supabase.table("users").insert({
        "username": username,
        "email": email,
        "password_hash": hashed_pw
    }).execute()
    return result

def login_user(email, password):
    try:
        result = supabase.table("users").select("*").eq("email", email).execute()
    except Exception as e:
        return None, f"Ошибка при запросе: {e}"

    if not result.data:
        return None, "Неверные данные для входа"

    user = result.data[0]  # Берём первого, если найдено несколько
    if verify_password(password, user["password_hash"]):
        return user["user_id"], None
    else:
        return None, "Неверные данные для входа"

