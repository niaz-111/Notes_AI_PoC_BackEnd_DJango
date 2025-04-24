from django.contrib import admin
from django.urls import path, include
from .views import *

urlpatterns = [
    path("load_note/", upload_note_with_file, name="upload_note_with_file"),
    path('ask_question/', ask_question, name='ask_question'),
    path('clear_notes/', clear_notes, name='clear_notes'),
    path('retrieval_chain_qa/', retrieval_chain_qa, name='retrieval_chain_qa'),
]
