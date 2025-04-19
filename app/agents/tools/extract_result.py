from app.model.ttt import TTT

from agents import function_tool

from typing import List
from pydantic import BaseModel

ttt = TTT()

class Message(BaseModel):
    role: str
    content: str

extract_result_json = {
            "type": "function",
            "name": "extract_result",
            "description": "Извлечь ТОЛЬКО элемент Result (Результат) из ПРЕДОСТАВЛЕННОЙ стенограммы собеседования. НЕ ДОБАВЛЯЙТЕ никакой информации, которой нет в тексте. НЕ ВЫДУМЫВАЙТЕ. Используйте только факты из стенограммы.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Result": {"type": "string", "description": "Извлеченный ДОСЛОВНО результат из собеседования. Если результат не описан ЯВНО, укажите 'Результат не описан'. Не делайте выводов и не обобщайте."}
                },
                "required": ["Result"],
                "additionalProperties": False
            },
            "strict": True,
        }

filter_conversation_json = {
            "type": "function",
            "name": "filter_conversation",
            "description": "Отфильтровать сообщения собеседования, оставив ТОЛЬКО те части, которые напрямую описывают Ситуацию, Задачу, Действия или Результат (STAR). Удалить всё остальное.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filtered_content": {"type": "boolean", "description": "Флаг, показывающий, остались ли сообщения, СТРОГО относящиеся к STAR. True - да, False - нет."},
                    "messages": {"type": "array", "items": {"type": "object", "properties": {"role": {"type": "string"}, "content": {"type": "string"}}}, "description": "Отфильтрованные сообщения, содержащие ТОЛЬКО информацию по STAR."}
                },
                "required": ["filtered_content", "messages"],
                "additionalProperties": False
            },
            "strict": True,
        }

def filter_conversation(messages: List[Message]) -> dict:
    """
    Использует агента для строгой фильтрации сообщений, оставляя только информацию по STAR.
    
    Args:
        messages (List[Message]): Исходный список сообщений
        
    Returns:
        dict: Словарь с отфильтрованными сообщениями и флагом наличия STAR информации
    """
    system_prompt = """Ты - агент-фильтр для текста собеседований по методологии STAR. Твоя ЗАДАЧА - ИСКЛЮЧИТЕЛЬНО фильтрация. Проанализируй КАЖДОЕ сообщение. ОСТАВЬ ТОЛЬКО те сообщения или ЧАСТИ сообщений, которые ЯВНО и НАПРЯМУЮ описывают один из элементов STAR: Ситуацию (Situation), Задачу (Task), Действие (Action), Результат (Result).

СТРОГО УДАЛИ:
1.  Приветствия (любые формы)
2.  Прощания (любые формы)
3.  Весь small talk (погода, как дела, общие фразы)
4.  Благодарности, извинения, комплименты
5.  Мета-комментарии о ходе беседы (например, "Хороший вопрос", "Давайте перейдем к следующему пункту")
6.  Вопросы интервьюера, если они не содержат описание элемента STAR (например, "Расскажите о себе", "Какие ваши сильные стороны?")
7.  Любые другие фразы, НЕ являющиеся прямым описанием S, T, A, или R.

НЕЛЬЗЯ:
-   Интерпретировать или обобщать.
-   Добавлять информацию.
-   Изменять формулировки.

Просто ВЕРНИ список сообщений, прошедших фильтр. Если после фильтрации НИ ОДНО сообщение не содержит ЯВНОЙ информации по STAR, установи флаг 'filtered_content' в False. В остальных случаях - True."""

    # Преобразовываем сообщения в формат, понятный для агента
    messages_for_agent = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Отфильтруй этот диалог строго по STAR. Удали всё лишнее."}
    ]
    
    # Добавляем сообщения собеседования
    for msg in messages:
        messages_for_agent.append({"role": msg.role, "content": msg.content})
    
    # Запускаем фильтрацию через агента
    response = ttt.generate_response_with_function(
        messages=messages_for_agent,
        functions=[filter_conversation_json]
    )
    
    return response

@function_tool
def extract_result(messages: List[Message]) -> str:
    """
    Извлечь элемент Result (Результат) из стенограммы собеседования. НЕ ВЫДУМЫВАТЬ. Использовать ТОЛЬКО текст стенограммы.

    Args:
        messages (list): Список сообщений из стенограммы собеседования. Каждое сообщение - словарь, содержащий 'role' и 'content'.

    Return:
        str: Извлеченный ДОСЛОВНО результат. Если результат не описан ЯВНО - 'Результат не описан'.
    """
    # Фильтруем сообщения от всего, кроме STAR
    filtered_result = filter_conversation(messages)
    
    # Проверяем, осталось ли достаточно содержательной информации
    if not filtered_result or not filtered_result.get("filtered_content", False):
        return "Недостаточно информации для вынесения решения."
    
    # Используем отфильтрованные сообщения для извлечения результата
    filtered_messages_data = filtered_result.get("messages", [])
    if not filtered_messages_data:
         return "Недостаточно информации для вынесения решения."

    filtered_messages = [Message(role=msg["role"], content=msg["content"]) 
                        for msg in filtered_messages_data]
    
    if not filtered_messages:
        return "Недостаточно информации для вынесения решения."
        
    # Добавляем системное сообщение перед вызовом экстрактора для усиления инструкций
    extraction_messages = [
        {"role": "system", "content": "Ты должен извлечь ТОЛЬКО Результат (Result) из предоставленного текста. Не добавляй ничего от себя. Не интерпретируй. Если Результат не описан явно, ВЕРНИ 'Результат не описан'."}
    ]
    for msg in filtered_messages:
         extraction_messages.append({"role": msg.role, "content": msg.content})

    response = ttt.generate_response_with_function(
        messages=extraction_messages, # Используем сообщения с добавленным системным промптом
        functions=[extract_result_json]
    )
    print("Response from extract_result:", response)
    # Дополнительная проверка ответа
    if isinstance(response, dict) and "Result" in response:
        extracted_result = response["Result"]
        if not extracted_result or extracted_result.strip() == "" or "не описан" in extracted_result.lower():
             # Если модель вернула пустую строку или указала на отсутствие, проверяем исходный текст
             combined_filtered_text = " ".join([msg.content for msg in filtered_messages])
             if not combined_filtered_text.strip(): # Если и отфильтрованный текст пуст
                 return "Недостаточно информации для вынесения решения."
             else:
                 return extracted_result # Возвращаем ответ модели (вероятно, 'Результат не описан')
        # Здесь можно добавить более сложную проверку на соответствие извлеченного текста исходному отфильтрованному тексту, если потребуется
        return extracted_result
    elif isinstance(response, str): # Если вернулась строка (например, ошибка или прямое сообщение)
        return response
        
    return "Ошибка извлечения информации."