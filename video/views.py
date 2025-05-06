import os
import uuid
import redis
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .tasks import process_uploaded_video
from dotenv import load_dotenv

load_dotenv()

# Initialize Redis
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))

class VideoUploadView(APIView):
    def post(self, request):
        uploaded_file = request.FILES.get("video")
        
        if not uploaded_file:
            return Response({"error": "No video file provided."}, status=400)

        video_id = str(uuid.uuid4())
        temp_video_path = f"temp_uploads/{video_id}.mp4"
        os.makedirs("temp_uploads", exist_ok=True)

        with open(temp_video_path, "wb+") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Mark as processing
        redis_client.setex(f"video_summary:{video_id}", 3600, "PROCESSING")

        # Start background task
        process_uploaded_video.delay(temp_video_path, video_id)

        return Response({"video_id": video_id}, status=202)


class VideoSummaryView(APIView):
    def get(self, request, video_id):
        key = f"video_summary:{video_id}"
        result = redis_client.get(key)
        print(result)

        if result is None:
            return Response({"error": "Invalid or expired video_id."}, status=404)

        result_str = result.decode("utf-8")
        if result_str == "PROCESSING":
            return Response({"status": "processing"}, status=202)

        if result_str.startswith("ERROR"):
            return Response({"status": "error", "message": result_str}, status=500)

        return Response({"status": "complete", "summary": result_str}, status=200)
