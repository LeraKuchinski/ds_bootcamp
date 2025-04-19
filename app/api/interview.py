from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import PlainTextResponse
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

# New HTTP endpoint for CV download
@router.get("/generate-cv-download", response_class=PlainTextResponse)
async def generate_cv_download(
    name: str = Query(..., description="Candidate Name"), 
    specialization: str = Query(..., description="Area of specialization"), 
    persona: str = Query(..., description="Target persona for the CV")
):
    """
    Generates (or retrieves from cache) a CV and returns it as a text file download.
    """
    try:
        # Use the existing caching mechanism
        cv_data = get_cached_cv(name=name, specialization=specialization, persona=persona)
        
        filename = cv_data.get("filename", "error_resume.txt")
        resume_content = cv_data.get("resume_content", "")

        # Check if content generation failed (based on the content string from cv_agent)
        if "Failed to generate resume" in resume_content or not resume_content:
            raise HTTPException(status_code=500, detail=f"Failed to generate or retrieve CV content: {resume_content}")

        # Prepare headers for file download
        headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
        
        # Return the resume content as a downloadable text file
        return PlainTextResponse(content=resume_content, media_type='text/plain', headers=headers)
        
    except Exception as e:
        # Catch potential errors during CV generation or retrieval
        print(f"Error in /generate-cv-download: {str(e)}")
        # Re-raise as HTTPException for FastAPI to handle
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
# Вебсокет-эндпоинт для интервью
@router.websocket("/ws/interview")
async def websocket_interview(ws: WebSocket, persona: str = Query("Junior Python Developer"), skill: str = Query("Python programming"), psyho_profile: str = Query("template_speaker")):
    await ws.accept()  # Принимаем подключение
    # системный промпт для агента на основе выбранной персоны и навыка
    # system_prompt = prompts["persona_system_prompt"].format(persona=persona, skill=skill)
    # Use cached CV instead of generating every time
    cv_data = get_cached_cv(name=persona, specialization=skill, persona=persona)
    cv_details = cv_data.get("resume_text", "No CV generated.")
    
    system_prompt = prompts['extended_persona_system_prompt']['template']#.format(persona=persona, skill=skill) # ["persona_system_prompt"]
    agent = create_interviewee_agent(system_prompt, psyho_profile, persona, skill, cv_details)  # агент для интервью
    try:
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
    except WebSocketDisconnect:
        pass