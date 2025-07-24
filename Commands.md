py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r .\requirements.txt
Download redis: https://github.com/tporadowski/redis/releases
Install: redis-server.exe

celery -A NotesChat_BackEnd_AI_POC worker --pool=threads --concurrency=4 --loglevel=info
python manage.py runserver 