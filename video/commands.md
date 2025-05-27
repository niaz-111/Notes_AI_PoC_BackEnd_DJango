python -m venv venv
.\venv\Scripts\activate
pip install opencv-python imagehash pillow numpy moviepy
pip install torch transformers pillow
.\venv\Scripts\python main.py
winget install Redis-x64
celery -A NotesChat_BackEnd_AI_POC worker -l info