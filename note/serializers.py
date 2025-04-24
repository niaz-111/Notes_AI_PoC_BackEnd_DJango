from rest_framework import serializers
from .models import Note

class NoteSerializer(serializers.ModelSerializer):
    class Meta:
        def __init__(self):
            model = Note
            fields = ['uid', 'title', 'text']