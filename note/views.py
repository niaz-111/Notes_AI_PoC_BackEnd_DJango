from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from .models import Note
from .serializers import NoteSerializer
from .pipeline_runner import run_ingestion_pipeline
from .vectorestore import delete_note_by_uid
from .qa_chain import get_conversational_rag_chain
from .vectorestore import get_vectorstore
from .qa_chain import get_retrieval_qa_chain

@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_note_with_file(request):
    uid = request.POST.get('uid')
    title = request.POST.get('title')
    text_file = request.FILES.get('text_file')
    text_content = request.POST.get('text_content')

    if not uid or not title:
        return Response({"error": "uid, title, and text_file are required."}, status=status.HTTP_400_BAD_REQUEST)

    if not text_file and not text_content:
        return Response({"error": "Either text_file or text_content is required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        if text_file:
            if not text_file.name.endswith('.txt'):
                return Response({"error": "Only .txt files are accepted."}, status=status.HTTP_400_BAD_REQUEST)
            text = text_file.read().decode('utf-8')
        else:
            text = text_content

        note_data = {
            "uid": uid,
            "title": title,
            "content": text,
        }

        delete_note_by_uid(uid)
        notes_list = []
        notes_list.append(note_data)

        run_ingestion_pipeline(notes_list)
        
        return Response("ok", status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    


# Placeholder for actual logic
# from .chat_engine import get_answer_from_langchain

@api_view(['POST'])
def ask_question(request):
    question = request.data.get('question')

    if not question:
        return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        conversational_rag_chain = get_conversational_rag_chain()

        answer = conversational_rag_chain.invoke(
            {"input": question},
            config={
            "configurable": {"session_id": "abc123"}
            },
            )["answer"]

        answer = f"Echoing: {question} — {answer}"

        return Response({"answer": answer}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@api_view(['GET'])
def clear_notes(request):
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()

        vectorstore.delete_collection()

        return Response({"message": f"Deleted {count} chunks."}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def retrieval_chain_qa(request):
    question = request.data.get('question')

    if not question:
        return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        qa_chain = get_retrieval_qa_chain()
        result = qa_chain({"query": question})

        return Response({"answer": result["result"]}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)