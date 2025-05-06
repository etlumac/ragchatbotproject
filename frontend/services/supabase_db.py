from supabase import create_client, Client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL") 
SUPABASE_KEY = os.getenv("SUPABASE_KEY") 

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# 🧠 CHAT HISTORY

def get_document_id(user_id: str, document_name: str):
    result = supabase.table("metadata")\
        .select("id")\
        .eq("user_id", user_id)\
        .eq("document_name", document_name)\
        .execute().data
    return result[0]["id"] if result else None

def fetch_chat_history(user_id):
    data = supabase.table("chat_history").select("messages, document_id").eq("user_id", user_id).execute().data

    # Получим отображение id -> имя
    doc_ids = [item["document_id"] for item in data]
    if not doc_ids:
        return {}

    docs = supabase.table("metadata").select("id, document_name").in_("id", doc_ids).execute().data
    id_to_name = {doc["id"]: doc["document_name"] for doc in docs}

    return {id_to_name[item["document_id"]]: item["messages"] for item in data if item["document_id"] in id_to_name}


def save_chat_history(user_id: str, document_name: str, messages: list):
    document_id = get_document_id(user_id, document_name)
    if not document_id:
        raise ValueError("Документ не найден для сохранения чата")

    existing = supabase.table("chat_history")\
        .select("id")\
        .eq("user_id", user_id)\
        .eq("document_id", document_id)\
        .execute().data

    if existing:
        supabase.table("chat_history")\
            .update({
                "messages": messages,
                "document_name": document_name  # на случай, если имя документа обновилось
            })\
            .eq("user_id", user_id)\
            .eq("document_id", document_id)\
            .execute()
    else:
        supabase.table("chat_history")\
            .insert({
                "user_id": user_id,
                "document_id": document_id,
                "document_name": document_name,  # ⚠️ обязательно, чтобы не было NULL
                "messages": messages
            }).execute()


def delete_chat_history(user_id: str, document_name: str):
    document_id = get_document_id(user_id, document_name)
    if not document_id:
        return
    supabase.table("chat_history")\
        .delete()\
        .eq("user_id", user_id)\
        .eq("document_id", document_id)\
        .execute()



# 📋 METADATA

def fetch_metadata(user_id):
    data = supabase.table("metadata").select("*").eq("user_id", user_id).execute().data
    return {item["document_name"]: item["data"] for item in data}


def save_metadata(user_id, document_name, data):
    existing = supabase.table("metadata").select("id").eq("user_id", user_id).eq("document_name", document_name).execute().data
    if existing:
        supabase.table("metadata").update({"data": data}).eq("user_id", user_id).eq("document_name", document_name).execute()
    else:
        supabase.table("metadata").insert({
            "user_id": user_id,
            "document_name": document_name,
            "data": data
        }).execute()

def delete_metadata(user_id, document_name):
    supabase.table("metadata").delete().eq("user_id", user_id).eq("document_name", document_name).execute()

