from supabase import create_client, Client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL") 
SUPABASE_KEY = os.getenv("SUPABASE_KEY") 

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# 🧠 CHAT HISTORY

def fetch_chat_history(user_id):
    data = supabase.table("chat_history").select("*").eq("user_id", user_id).execute().data
    return {item["document_name"]: item["messages"] for item in data}

def save_chat_history(user_id: str, document_name: str, messages: list):
    # Проверяем, есть ли уже история для этого пользователя и документа
    existing = supabase.table("chat_history")\
        .select("id")\
        .eq("user_id", user_id)\
        .eq("document_name", document_name)\
        .execute().data

    if existing:
        supabase.table("chat_history")\
            .update({"messages": messages})\
            .eq("user_id", user_id)\
            .eq("document_name", document_name)\
            .execute()
    else:
        supabase.table("chat_history")\
            .insert({
                "user_id": user_id,
                "document_name": document_name,
                "messages": messages
            }).execute()


def delete_chat_history(user_id: str, document_name: str):
    supabase.table("chat_history")\
        .delete()\
        .eq("user_id", user_id)\
        .eq("document_name", document_name)\
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

