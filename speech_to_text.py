# pylint: disable=broad-exception-caught, global-statement, invalid-name, line-too-long, missing-function-docstring, multiple-statements
'''Module Docstring'''

from asyncio import run as asyncio_run
from json import load as json_load, loads as json_loads
from multiprocessing import Process, Queue
from threading import Thread
from typing import List
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from Levenshtein import distance as levenshtein_distance
import pyaudio # type: ignore
import uvicorn
import vosk # type: ignore

app = FastAPI()

recognized_text = ""



def load_config(config_path: str):
    """Load configuration from a JSON file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return json_load(file)

config = load_config("keyword_recognition_config.json")


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self): self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket): self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Sends a message to all connected WebSocket clients."""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)


manager = ConnectionManager()
command_history: dict[str, float] = {}


def single_word_fuzzy_regex(word, partial=False):
    """Fuzzy match a single word against predefined commands with group-specific tolerances."""

    # Command groups with associated tolerances
    command_mapping = {
        "movement": {
            "right" : "r",
            "left"  : "l",
            "jump"  : "j",
            "down"  : "d",
        },
        "menu": {
            "pause" : "pause",
            "continue" : "continue",
            "restart" : "restart",
            "quit" : "quit",
        },
        "mode": {
            "slow" : "slow",
            "normal" : "normal",
            "fast" : "fast",
        },
    }

    tolerances = {
        "movement" : 5,
        "menu" : 2,
        "mode" : 3,
    }

    current_time = time.time()
    cooldown_time = 0.4 if partial else 0.6

    flattened_mapping = {
        cmd: (group, output)
        for group, commands in command_mapping.items()
        for cmd, output in commands.items()
    }

    best_match = min(flattened_mapping.keys(), key=lambda cmd: levenshtein_distance(word, cmd))
    distance = levenshtein_distance(word, best_match)

    group, user_input = flattened_mapping[best_match]
    tolerance = tolerances[group]

    if distance <= tolerance:
        # Prevent duplicates / Skip duplicate commands
        if best_match in command_history:
            last_time = command_history[best_match]
            if current_time - last_time < cooldown_time:
                return ""

        command_history[best_match] = current_time
        print(f"Fuzzy result: {user_input}")
        return user_input

    return ""


def audio_capture(audio_queue):
    """Capture audio and send it to the queue."""
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=config["general"]["sample_rate"],
        input=True,
        frames_per_buffer=config["pyaudio"]["frames_per_buffer"]
    )
    stream.start_stream()

    print("Listening for speech...")
    while True:
        audio_data = stream.read(config["pyaudio"]["data_chunk_size"], exception_on_overflow=False)
        audio_queue.put(audio_data)


def vosk_recognition(audio_queue, result_queue):
    """Recognize speech from audio data in the queue."""
    global recognized_text

    recognizer = vosk.KaldiRecognizer(
        vosk.Model(config["vosk"]["vosk_model_path"]),
        config["general"]["sample_rate"]
    )

    while True:
        audio_data = audio_queue.get()
        if recognizer.AcceptWaveform(audio_data):
            recognized_text = json_loads(recognizer.Result())["text"]
            print(f"Full result: {recognized_text}")
            result_queue.put(single_word_fuzzy_regex(recognized_text))
        else:
            partial_text = json_loads(recognizer.PartialResult())["partial"]
            if partial_text:
                last_word = partial_text.split()[-1]
                print(f"Partial result (last word): {last_word}")
                result_queue.put(single_word_fuzzy_regex(last_word, partial=True))



# RESTful Endpoints
@app.get('/get_text', response_class=PlainTextResponse)
async def get_text():
    """Endpoint to retrieve the recognized text as plain text."""
    return recognized_text


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)



# MAIN
if __name__ == '__main__':
    queue_audio:  Queue = Queue()
    queue_result: Queue = Queue()

    audio_process = Process(target=audio_capture, args=(queue_audio,))
    recognition_process = Process(target=vosk_recognition, args=(queue_audio, queue_result))

    audio_process.start()
    recognition_process.start()

    def result_broadcast_loop():
        while True:
            if not queue_result.empty():
                command = queue_result.get()
                if command:
                    asyncio_run(manager.broadcast(command))

    broadcast_thread = Thread(target=result_broadcast_loop)
    broadcast_thread.daemon = True
    broadcast_thread.start()

    uvicorn.run(app, host='0.0.0.0', port=5000)

    audio_process.join()
    recognition_process.join()
