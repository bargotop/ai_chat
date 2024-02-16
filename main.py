import asyncio
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import StreamingResponse, Response

from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, conint
from fastapi.middleware.cors import CORSMiddleware

from tools.tts import UlutTTS


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
OPENAI_IS_STREAM = eval(os.getenv("OPENAI_IS_STREAM"))

client = OpenAI(
    api_key=OPENAI_API_KEY
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Можете указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/chunked')
async def chunked_func():
    result = generate_chunks()
    return StreamingResponse(result, media_type="text/plain")


async def generate_chunks():
    for i in range(5):
        yield f"Chunk {i}\n"
        await asyncio.sleep(0.5)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()

        if data:
            async for part_result in generate_streamed(data, True):
                await websocket.send_text(part_result)


class TTSRequest(BaseModel):
    text: str
    speaker_id: conint(ge=1, le=2)


@app.post('/tts')
async def text_to_speech(request: TTSRequest):
    entity = UlutTTS(
        text=request.text,
        speaker_id=request.speaker_id
    )
    result = entity.to_speech()
    return Response(content=result, media_type="audio/mpeg")


class UserRequest(BaseModel):
    text: str
    is_stream: bool


@app.post('/ask')
async def ask(request: UserRequest):
    try:
        if request.is_stream:
            result = generate_streamed(request.text, request.is_stream)
            return StreamingResponse(result)
        else:
            result = await generate(request.text, request.is_stream)
            return {
                'data': {
                    'answer': result
                }
            }

    except Exception as e:
        print(f"Exception: {e=}")
        raise HTTPException(status_code=500, detail=str(e))


@app.options("/ask", status_code=200)
def options_ask():
    return {"method": "OPTIONS"}


async def generate_streamed(text: str, is_stream: bool):
    response = await get_chat_response(text, is_stream)
    for chunk in response:
        chunk: ChatCompletionChunk
        if chunk.choices[0].finish_reason is None:
            part_result = chunk.choices[0].delta.content
            yield part_result
        else:
            yield ''


async def generate(text: str, is_stream: bool):
    response = await get_chat_response(text, is_stream)
    result = response.choices[0].message.content
    return result


async def get_chat_response(text: str, is_stream: bool):
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {'role': 'user', 'content': text}
        ],
        stream=is_stream
    )
    return response

