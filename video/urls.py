from django.urls import path
from .views import VideoUploadView, VideoSummaryView

urlpatterns = [
    path('upload', VideoUploadView.as_view(), name='upload-video'),
    path('summary/<uuid:video_id>', VideoSummaryView.as_view(), name='video-summary'),
]
