from celery import shared_task
from .rag.ingestion import process_pdf
from django.contrib.sessions.models import Session
import os

@shared_task
def process_pdf_task(file_path, session_id):
    try:
        # Clear chat history for this session so the old document conversation is erased
        sessions = Session.objects.all()
        for session_obj in sessions:
            session_data = session_obj.get_decoded()
            if session_data.get('custom_session_id') == session_id:
                chat_history_key = f'chat_history_{session_id}'
                if chat_history_key in session_data:
                    session_data[chat_history_key] = []
                    session_obj.session_data = Session.objects.encode(session_data)
                    session_obj.save()
                break
                
        # Process the file via RAG components
        with open(file_path, 'rb') as f:
            chunk_count = process_pdf(f, session_id)
            
        return {"status": "success", "chunks_processed": chunk_count}
        
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    finally:
        # Erase the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)