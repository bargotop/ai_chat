import os
import requests
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

ULUT_TTS_API_KEY = os.getenv('ULUT_TTS_API_KEY')
ULUT_TTS_API_URL = os.getenv('ULUT_TTS_API_URL')


class UlutTTS:
    text: str
    speaker_id: int
    _output_path: str = '../output/{}.mp3'

    def __init__(self, text: str, speaker_id: int) -> None:
        if speaker_id not in (1, 2):
            raise ValueError("speaker_id должен быть 1 или 2")

        self.text = text
        self.speaker_id = speaker_id

    def to_speech(self) -> bytes:
        data = {
            'text': self.text,
            'speaker_id': self.speaker_id
        }

        response = requests.post(
            url=ULUT_TTS_API_URL,
            json=data,
            headers={'Authorization': f'Bearer {ULUT_TTS_API_KEY}'},
            verify=False
        )

        result = response.iter_content(100)

        return response.content

    def _save_file(self, binary: bytes) -> str:
        file_name = self._output_path.format(uuid4().hex)

        with open(file_name, 'wb') as file:
            file.write(binary)

        return file_name


