from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import base64
import json
import os

from app.model.ttt import TTT
from app.model.stt import STT
from app.model.tts import TTS
from app.agents.prompts.utils import load_prompts
from pprint import pp
from agents import Runner
from app.agents.interviewee_agent import create_interviewee_agent
from cv_agent import generate_cv

from app.agents.profiles import candidate_profiles

# Используем директорию /tmp для временных файлов (доступна для записи всем пользователям)
TEMP_DIR = "/tmp/ai-interview-temp"
os.makedirs(TEMP_DIR, exist_ok=True)

router = APIRouter()

ttt = TTT()
stt = STT()
tts = TTS()

prompts = load_prompts("persona_system_prompt.yaml")

# Simple CV cache to avoid regenerating CVs for the same parameters
cv_cache = {}

# Function to get CV with caching
def get_cached_cv(name: str, specialization: str, persona: str):
    # Create a cache key from the parameters
    cache_key = f"{name}_{specialization}_{persona}"
    
    # Check if we already have this CV in cache
    if cache_key in cv_cache:
        print(f"Using cached CV for {cache_key}")
        return cv_cache[cache_key]
    
    # If not in cache, generate and store it
    print(f"Generating new CV for {cache_key}")
    cv_data = generate_cv(name=name, specialization=specialization, persona=persona)
    cv_cache[cache_key] = cv_data
    
    return cv_data

# Вебсокет-эндпоинт для интервью
@router.websocket("/ws/interview")
async def websocket_interview(ws: WebSocket, name: str = Query("Candidate"), persona: str = Query("Junior Python Developer"), skill: str = Query("Python programming"), psyho_profile: str = Query("template_speaker")):
    
    await ws.accept()  # Принимаем подключение

    # Use cached CV instead of generating every time
    cv_data = get_cached_cv(name=name, specialization=skill, persona=persona)
    cv_details = cv_data.get("resume_text", "No CV generated.")
    
    # 2. Load and format the system prompt including CV details
    system_prompt_template = prompts['extended_persona_system_prompt']['template']
    system_prompt = system_prompt_template.format(
        persona=persona, 
        skill=skill, 
        name=name, 
        cv_details=cv_details
    )

    # 3. Create the agent with the enhanced prompt
    agent = create_interviewee_agent(system_prompt, psyho_profile, persona, skill)  # агент для интервью

    # --- End Setup ---

    try:
        # --- Message Handling Loop ---
        while True:
            data = await ws.receive_text()  # сообщение от клиента
            json_data = json.loads(data)
            if json_data["type"] == "text":  # текст
                user_input = json_data.get("message", "")
                is_audio = False
            elif json_data["type"] == "audio":  # аудио
                audio_bytes = base64.b64decode(json_data["audio"])
                temp_audio_path = os.path.join(TEMP_DIR, "temp_audio.wav")
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_bytes)  # Сохраняем аудио во временный файл
                user_input = stt.transcribe_from_path(temp_audio_path)  # Распознаём речь
                is_audio = True
            # Формируем историю сообщений для передачи агенту
            messages = [ttt.create_chat_message(msg["role"], msg["content"]) for msg in json_data.get("history", [])]
            messages.append(ttt.create_chat_message("user", user_input))  # Добавляем текущее сообщение пользователя
            # Получаем ответ от агента
            response = await Runner.run(agent, messages)
            # response = await Runner.run(agent, user_input, context={"messages": messages}) # Вариант с контекстом
            agent_text = response.final_output  # Текстовый ответ агента
            if is_audio:
                # Генерируем аудиофайл с ответом агента
                tts_response = tts.generate_speech(agent_text, tone=prompts["persona_voice_tone_prompt"])
                agent_audio = base64.b64encode(tts_response.content).decode('utf-8')
                # Отправляем клиенту текст и аудио
                await ws.send_json({"type": "voice", "content": agent_text, "user_text": user_input, "audio": agent_audio})
            elif not is_audio:
                # Отправляем клиенту только текст
                await ws.send_json({"type": "text", "content": agent_text})
        # --- End Message Handling Loop ---
    except WebSocketDisconnect:
        pass