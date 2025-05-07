from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from qdrant_client import QdrantClient
import os
import json
import traceback
from dotenv import load_dotenv
from slugify import slugify
from typing import List, Tuple
from langchain_qdrant import Qdrant
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("❌ Не хватает переменных окружения: проверь .env")

READER_MODEL_NAME = "deepseek/deepseek-chat"
COLLECTION_NAME = "documents"
METADATA_PATH = "static/metadata.json"


client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.vsegpt.ru/v1"
)

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    document_name: str
    user_id: str 

def safe_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    return f"{slugify(name)}{ext}"

def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def generate_multi_queries(question: str) -> List[str]:
    try:
        prompt = (
            f"Перефразируй следующий вопрос не более чем 2 способавми, сохраняя его смысл: '{question}'. "
            "Постарайся сделать это так, чтобы каждый перефразированный вопрос не повторял слова оригинального, но всё ещё сохранял исходную идею, рассматривал вопрос с другой стороны.Убери из вопроса лишнее и повысь плотность ключевых слов"
        )

        messages = [
            {
                "role": "system",
                "content": "Ты помогаешь пользователям создавать различные варианты вопросов на основе заданного. Пожалуйста, перефразируй вопрос несколькими способами."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        print("🤖 Отправляем запрос для перефразирования вопроса...")
        completion = client.chat.completions.create(
            model=READER_MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )

        generated_queries = completion.choices[0].message.content.strip().split("\n")
        print(f"✅ Сгенерированные перефразированные вопросы: {generated_queries}")
        return generated_queries

    except Exception:
        print("❌ Ошибка при генерации перефразированных вопросов:", traceback.format_exc())
        return [question]

def answer_with_rag_multi_query(
    question: str,
    document_name: str,
    user_id: str,  # 👈 добавили
    qdrant_client: QdrantClient,
    embeddings: OpenAIEmbeddings
) -> Tuple[str, List[str]]:

    print(f"🔍 Исходный вопрос: {question}")
    document_filter = safe_filename(document_name)

    multi_queries = generate_multi_queries(question)
    all_relevant_docs = []

    filter_condition = Filter(
    must=[
        FieldCondition(
            key="document_id",
            match=MatchValue(value=document_filter)
        ),
        FieldCondition(
            key="user_id",  # 👈 новый фильтр
            match=MatchValue(value=user_id)
        )
    ]
)


    try:
        # === 1. Классический поиск по перефразировкам ===
        for query in multi_queries:
            print(f"🔎 Поиск по перефразировке: '{query}'")
            query_vector = embeddings.embed_query(query)
            hits = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=10,
                query_filter=filter_condition
            )
            all_relevant_docs.extend([h.payload.get("text", "") for h in hits])

        # === 2. HyDE: генерируем гипотетический ответ ===
        print("🧠 Генерируем гипотетический ответ (HyDE)...")
        hyde_prompt = [
            {
                "role": "system",
                "content": (
                    "Ты ассистент, который должен попытаться ответить на вопрос пользователя "
                    "максимально правдоподобно, даже если ты не уверен в точности ответа. "
                    "Просто сгенерируй гипотетический ответ, опираясь на общие знания."
                )
            },
            {
                "role": "user",
                "content": f"{question}"
            }
        ]
        hyde_completion = client.chat.completions.create(
            model=READER_MODEL_NAME,
            messages=hyde_prompt,
            temperature=0.7,
            max_tokens=300,
        )
        hyde_answer = hyde_completion.choices[0].message.content.strip()
        print(f"📄 HyDE ответ: {hyde_answer[:200]}...")

        # Встраиваем HyDE-ответ
        hyde_vector = embeddings.embed_query(hyde_answer)

        # Поиск по вектору HyDE
        print("🔎 Поиск по HyDE-вектору...")
        hyde_hits = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=hyde_vector,
            limit=10,
            query_filter=filter_condition
        )
        all_relevant_docs.extend([h.payload.get("text", "") for h in hyde_hits])

        if not all_relevant_docs:
            return "Ответ не найден в загруженных документах.", []

        # Финальный контекст (можно добавить удаление дублей)
        context = "\n".join(list(dict.fromkeys(all_relevant_docs)))  # remove duplicates, keep order
        print(f"📚 Контекст после объединения: {len(context)} символов")

        # === Генерация финального ответа ===
        messages = [
            {
                "role": "system",
                "content": (
                    "Вы — аналитическая модель, отвечающая на вопросы на основе предоставленного контекста. "
                    "Ваш ответ должен быть чётким, точным и строго основанным на информации из контекста. "
                    "Перед тем как дать окончательный ответ, выполните внутреннее рассуждение, чтобы обоснованно прийти к выводу.\n\n"

                    "Инструкции для рассуждения:\n"
                    "1. Определите, какая информация требуется для ответа.\n"
                    "2. Найдите возможные соответствия в контексте.\n"
                    "3. Проверьте, действительно ли они соответствуют вопросу, а не просто похожи.\n"
                    "4. На основе анализа дайте точный ответ или сообщите, что информации недостаточно.\n\n"
                    
                    "Формат ответа:\n"
                    "Краткий и точный ответ на вопрос. Если информации недостаточно — прямо укажите это.\n\n"
                    "Промежуточные шаги рассуждения: что искали, где нашли, почему это релевантно. Уделите внимание сопоставимости данных из контекста с запросом. "
                    "Укажи источник откуда была взята информация в виде /\"Источник:/\ *тут укажи конкретную цитату из контекста откуда взята информацию*"
                    "Избегайте подмены похожих чисел или фактов. Не делайте предположений.\n\n"

                )
            },
            {
                "role": "user",
                "content": f"Контекст:\n{context}\n---\nВопрос: {question}"
            }
        ]

        print("🤖 Отправляем запрос в LLM...")
        completion = client.chat.completions.create(
            model=READER_MODEL_NAME,
            messages=messages,
            temperature=0.1,
        )

        answer = completion.choices[0].message.content.strip()
        return answer, all_relevant_docs

    except Exception:
        print("❌ Ошибка в `answer_with_rag_multi_query`:", traceback.format_exc())
        return "Ошибка обработки запроса.", []



@router.post("/chat/")
async def chat_with_rag(request: ChatRequest):
    query = request.query
    document_name = request.document_name
    user_id = request.user_id
    print(f"📩 Запрос: {query} (Документ: {document_name})")

    try:
        embeddings = OpenAIEmbeddings(
            model="emb-openai/text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base="https://api.vsegpt.ru/v1",
        )

        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        print(f"📥 Подготовка к поиску в коллекции `{COLLECTION_NAME}` с фильтром `{document_name}`...")

        answer, sources = answer_with_rag_multi_query(query, document_name, user_id, qdrant_client, embeddings)

        print(f"✅ Ответ: {answer}")
        return {"response": answer, "sources": sources}

    except HTTPException as http_ex:
        raise http_ex
    except Exception:
        print("❌ Ошибка в `chat_with_rag`:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса.")


