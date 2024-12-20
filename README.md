# SpeechToText

## Overview

This script implements a speech recognition service that listens for audio input and provides recognized text through a FastAPI-based web server. It uses a speech recognition engine (default: Vosk) to continuously process speech from the microphone and updates the recognized text in real time.

The FastAPI server offers multiple endpoints for connectivity, including WebSocket support for real-time updates to connected clients.

The script runs the speech recognition loop in a separate thread, allowing it to continuously capture speech while the server handles API requests and WebSocket connections concurrently.

## Instructions

**Warning:** Do not use with `Python 3.13`, as most plugins are not yet compatible. Instead, use `Python 3.11` or `3.12`.  
  I personally use `3.11.3` at the moment.

### How to run

... if you have multiple python installations (and the default installation is not `3.11`):

- In your terminal locate the `Python311` folder.  
  For me it's: `C:\Users\Admin\AppData\Local\Programs\Python\Python311`

**Make sure the terminal runs on this folder!**

- In the terminal run: `python.exe -m pip install RequiredPackageNameHere`  
  Refer to the package section below.

- In `VSCode` or your editor of choice open a terminal (doesn't need to be on the `Python311` folder path.)  
  and run: `py -V:3.11 .\speech_to_text.py`

---

**When using Vosk** for speech recognition, copy a Vosk model (folder) from the `voskModels` folder (e.g. `vosk-model-en-us-0.22`) and paste it into the projects main folder (e.g. have it sit next to the `speech_to_text.py` script).

Delete the current `model` folder and rename the Vosk model folder you've just choosen to `model`.

You can download additional Vosk models from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models).

**When using TensorFlow** for speech recognition, make sure to use a compatible TensorFlow version, e.g. `1.x` (`pip install tensorflow==1.15.5`) instead of `2.x` (`pip install tensorflow`). This is required because the `speech_recognition` package relies on an older TensorFlow API (from TensorFlow `1.x`)

### Required Packages

**Note: Not all packages are required, depending on the setup!**

When installing this projects core package ...

- **[SpeechRecognition](https://pypi.org/project/SpeechRecognition/)**  
  Library for performing speech recognition, with support for several engines and APIs, online and offline.

... `pip` will automatically download most dependencies, however there are a few additional packages that are not fetched by `pip` and are required for the script to function properly.

The following packages are not automatically installed by `pip` and need to be installed through pip manually:

- **[PyAudio](https://pypi.org/project/PyAudio/)**  
  PyAudio provides Python bindings for PortAudio v19, the cross-platform audio I/O library.

- **[soundfile](https://pypi.org/project/soundfile/)**  
  Library for reading and writing sound files.

- **[uvicorn](https://pypi.org/project/uvicorn/)**  
  Uvicorn is an ASGI web server implementation for Python.

- **[fastapi](https://pypi.org/project/fastapi/)**  
  FastAPI is a modern, high-performance, web framework for building APIs with Python based on standard Python type hints.

---

Depending on the model you want to use, not all of these packages may be required. Select the ones you need.

- **[vosk](https://pypi.org/project/vosk/)** (Recommended!)  
  Vosk is an offline open source speech recognition toolkit.

- **[tensorflow](https://pypi.org/project/tensorflow/)**  
  TensorFlow is an open source software library for high performance numerical computation.

- **openai-whisper**

  - **[torch](https://pypi.org/project/torch/)**  
    PyTorch, a deep learning framework.

  - **[whisper](https://pypi.org/project/whisper/)** (Not sure if required!)  
    A text-to-speech library.

  - **[openai-whisper](https://pypi.org/project/openai-whisper/)**  
    OpenAI's Whisper model for speech-to-text transcription.

## Sources

### Speech recognition in python (great for general learning)

- [https://realpython.com/python-speech-recognition/](https://realpython.com/python-speech-recognition/)
- [https://www.simplilearn.com/tutorials/python-tutorial/speech-recognition-in-python](https://www.simplilearn.com/tutorials/python-tutorial/speech-recognition-in-python)
- [https://medium.com/naukri-engineering/speech-recognition-in-python-fcda027a97a1](https://medium.com/naukri-engineering/speech-recognition-in-python-fcda027a97a1)
