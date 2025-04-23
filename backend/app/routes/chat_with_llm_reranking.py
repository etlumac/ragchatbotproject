from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os
import json
import traceback
from dotenv import load_dotenv
from pathlib import Path
import re
from slugify import slugify
from typing import List

# Загружаем переменные из .env
load_dotenv()

# Читаем API-ключ из переменной окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Ошибка: API-ключ не найден! Укажите его в .env или в переменной окружения.")

# Пути
VECTOR_DB_PATH = Path("static/vectorstore")
METADATA_PATH = Path("static/metadata.json")  # Файл с метаинформацией о документах
READER_MODEL_NAME = "deepseek/deepseek-chat"

# Инициализируем OpenAI клиент
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.vsegpt.ru/v1",
)

# Инициализируем FastAPI роутер
router = APIRouter()

# ✅ Описание структуры запроса
class ChatRequest(BaseModel):
    query: str
    document_name: str  # Добавляем имя документа для поиска

def safe_filename(filename: str) -> str:
    """Приводит имя файла к единому безопасному формату."""
    name, ext = os.path.splitext(filename)
    safe_name = slugify(name, allow_unicode=False)  # Используем ASCII-only slugify
    return f"{safe_name}{ext}"

def load_metadata():
    """Загружает метаинформацию о документах."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def find_vector_store(document_name: str) -> Path:
    """Ищет FAISS-хранилище, сначала проверяя `metadata.json`, затем через `safe_filename`."""

    # Конвертируем имя файла в безопасный формат
    document_name_safe = safe_filename(document_name)
    
    # Загружаем метаданные
    metadata = load_metadata()

    # Проверяем путь в metadata.json
    if document_name_safe in metadata and "vector_index" in metadata[document_name_safe]:
        vector_index_path = Path(metadata[document_name_safe]["vector_index"])
        if vector_index_path.exists() and vector_index_path.is_dir():
            return vector_index_path

    # Если пути нет в metadata.json, проверяем стандартный путь
    vector_index_path = VECTOR_DB_PATH / f"{document_name_safe}.faiss"
    if vector_index_path.exists() and vector_index_path.is_dir():
        return vector_index_path

    raise HTTPException(status_code=500, detail=f"❌ Векторное хранилище `{document_name_safe}` отсутствует. Загрузите документ снова.")


def rerank_documents_with_llm_and_vector(
    question: str,
    vector_scored_docs: List[tuple],  # (Document, vector_score)
    client: OpenAI,
    vector_weight: float = 0.3,
    llm_weight: float = 0.7
) -> List[tuple]:
    """
    Выполняет LLM-реранкинг документов с учётом векторного скора.
    Возвращает список: (документ, финальный_скор)
    """
    reranked_docs = []

    # Нормализация vector_score (FAISS similarity) до [0, 1]
    scores = [score for _, score in vector_scored_docs]
    max_score, min_score = max(scores), min(scores)
    score_range = max_score - min_score if max_score != min_score else 1.0

    for doc, vector_score in vector_scored_docs:
        try:
            norm_vector_score = (vector_score - min_score) / score_range

            prompt = generate_rerank_prompt(doc.page_content, question)

            messages = [
                {"role": "system", "content": "Вы аналитическая модель, оценивающая релевантность текста для ответа на вопрос."},
                {"role": "user", "content": prompt}
            ]

            completion = client.chat.completions.create(
                model=READER_MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=50,
            )

            llm_score = float(re.findall(r"0(?:\.\d+)?|1(?:\.0+)?", completion.choices[0].message.content)[0])
            final_score = (norm_vector_score * vector_weight) + (llm_score * llm_weight)

            reranked_docs.append((doc.page_content, final_score))
            print(f"📊 norm_vector: {norm_vector_score:.2f}, llm: {llm_score:.2f}, final: {final_score:.3f}")

        except Exception as e:
            print("❌ Ошибка при реранкинге:", traceback.format_exc())
            reranked_docs.append((doc.page_content, 0.0))  # fallback

    return sorted(reranked_docs, key=lambda x: x[1], reverse=True)

def generate_rerank_prompt(document_text: str, question: str) -> str:
    return f"""
    Вам нужно оценить, насколько данный текст полезен для ответа на вопрос. Ответьте на вопрос, указав степень релевантности текста на шкале от 0 до 1, где:

    0 = Абсолютно нерелевантный: текст не имеет отношения к запросу.
    0.1 = Практически нерелевантный: только слабая или косвенная связь с запросом.
    0.2 = Очень слабо релевантный: текст содержит минимальную связь с запросом.
    0.3 = Слабо релевантный: связь слабая, но есть хотя бы один аспект.
    0.4 = Умеренно релевантный: текст содержит несколько аспектов, которые могут быть полезны для ответа.
    0.5 = Средне релевантный: текст помогает, но требует дополнительной информации.
    0.6 = Довольно релевантный: текст предоставляет полезную информацию для ответа.
    0.7 = Очень релевантный: текст непосредственно помогает в ответе на запрос.
    0.8 = Почти полностью релевантный: текст имеет значительное отношение к запросу.
    0.9 = Почти идеально релевантный: текст идеально подходит для ответа.
    1.0 = Абсолютно релевантный: текст содержит всю необходимую информацию для точного ответа.

    Текст документа:
    "{document_text}"

    Вопрос:
    "{question}"

    Оцените степень релевантности от 0 до 1.
    """


# Функция для генерации вариантов запросов
def generate_multi_queries(question: str) -> List[str]:
    """Генерирует несколько перефразированных вариантов запроса с использованием модели GPT."""
    try:
        # Создаём запрос для модели, чтобы она перефразировала вопрос
        prompt = (
            f"Перефразируй следующий вопрос одним способом  (не больше!!), сохраняя его смысл: '{question}'. "
            "Постарайся сделать это так, чтобы каждый перефразированный вопрос был уникальным, но всё ещё сохранял исходную идею."
        )

        # Отправляем запрос в модель для генерации перефразированных вопросов
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

        # Извлекаем сгенерированные перефразированные вопросы из ответа
        generated_queries = completion.choices[0].message.content.strip().split("\n")
        print(f"✅ Сгенерированные перефразированные вопросы: {generated_queries}")
        
        return generated_queries

    except Exception as e:
        print("❌ Ошибка при генерации перефразированных вопросов:", traceback.format_exc())
        return [question]  # Если ошибка, возвращаем исходный вопрос









def answer_with_rag_multi_query(question: str, knowledge_index: FAISS, client: OpenAI):
    print(f"🔍 Запросы: {question}")
    
    multi_queries = generate_multi_queries(question)
    combined_scored_docs = []

    try:
        for query in multi_queries:
            relevant_docs = knowledge_index.similarity_search_with_score(query=query, k=5)
            combined_scored_docs.extend(relevant_docs)

        if not combined_scored_docs:
            return "Ответ не найден в загруженных документах.", []

        # Реранкинг с LLM и векторной нормализацией
        reranked_docs = rerank_documents_with_llm_and_vector(
            question=question,
            vector_scored_docs=combined_scored_docs,
            client=client
        )

        # Формируем итоговый контекст (топ-N документов)
        top_contexts = [doc for doc, _ in reranked_docs[:5]]
        context = "\n".join(top_contexts)

        messages = [
    {
        "role": "system",
        "content": (
            "Вы — аналитическая модель, отвечающая на вопросы на основе предоставленного контекста. "
            "Ваш ответ должен быть чётким, точным и строго основанным на информации из контекста. "
            "Перед тем как дать окончательный ответ, выполните внутреннее рассуждение (Chain of Thought), чтобы обоснованно прийти к выводу.\n\n"

            "Формат ответа:\n"
            "---\n"
            "**Ответ:**\n"
            "[Краткий и точный ответ на вопрос. Если информации недостаточно — прямо укажите это.]\n\n"
            "**Обоснование:**\n"
            "[Промежуточные шаги рассуждения: что искали, где нашли, почему это релевантно. Уделите внимание сопоставимости данных из контекста с запросом. "
            "Избегайте подмены похожих метрик или фактов. Не делайте предположений.]\n\n"
            "**Вывод:**\n"
            "[(Опционально) Финальное обобщение, если требуется дополнительно пояснить суть или уровень уверенности в ответе.]\n"
            "---\n\n"

            "Инструкции для рассуждения:\n"
            "1. Определите, какая информация требуется для ответа.\n"
            "2. Найдите возможные соответствия в контексте.\n"
            "3. Проверьте, действительно ли они соответствуют вопросу, а не просто похожи.\n"
            "4. На основе анализа дайте точный ответ или сообщите, что информации недостаточно.\n\n"
            "5. Не ссылайтесь на названия или номера документов.Не указывайте источники в любом виде.\n"


            "Будьте особенно внимательны при наличии в контексте частичных или схожих по смыслу данных. Не подменяйте их целевыми. Не указывай в ответе источник информации."
        )
    },
    {
        "role": "user",
        "content": f"Context:\n{context}\n---\nQuestion: {question}"
    }
]

        completion = client.chat.completions.create(
            model=READER_MODEL_NAME,
            messages=messages,
            temperature=0.1,
        )

        answer = completion.choices[0].message.content
        print(f"✅ Ответ GPT: {answer}")
        return answer, top_contexts

    except Exception as e:
        print("❌ Ошибка:", traceback.format_exc())
        return "Ошибка обработки запроса.", []













@router.post("/chat/")
async def chat_with_rag(request: ChatRequest):
    """Обработчик чата с MultiQuery: принимает запрос, ищет в FAISS по нескольким запросам и отвечает через LLM."""
    query = request.query
    document_name = request.document_name
    print(f"📩 Получен запрос: {query} (Документ: {document_name})")

    try:
        # Загружаем FAISS-хранилище
        vectorstore_folder = find_vector_store(document_name)

        embeddings = OpenAIEmbeddings(
            model="emb-openai/text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base="https://api.vsegpt.ru/v1",
        )

        print(f"📥 Загружаем FAISS из `{vectorstore_folder}`...")
        vector_db = FAISS.load_local(str(vectorstore_folder), embeddings, allow_dangerous_deserialization=True)

        # Генерация ответа с MultiQuery
        answer, sources = answer_with_rag_multi_query(query, vector_db, client)


        print(f"✅ Ответ отправлен: {answer}")
        return {"response": answer, "sources": sources}

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print("❌ Ошибка в `chat_with_rag_multi_query`:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса.")