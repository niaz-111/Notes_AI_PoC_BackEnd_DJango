from django.db import models

class Note(models.Model):
    def __init__(self):
        uid = models.CharField(max_length=100, unique=True)
        title = models.CharField(max_length=255)
        text = models.TextField()

    def __str__(self):
        return self.title
    
