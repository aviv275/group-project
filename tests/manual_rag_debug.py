import os
import sys
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.rag_utils import create_rag_system

if __name__ == "__main__":
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("No GOOGLE_API_KEY set. Please export it and rerun.")
        sys.exit(1)
    
    # Force creation of new vector store by removing existing files
    vector_store_path = "models/rag_system/vector_store.index"
    if os.path.exists(vector_store_path):
        shutil.rmtree(vector_store_path)
        print(f"Removed existing vector store: {vector_store_path}")
    
    print("Creating new RAG system with Google API key...")
    rag = create_rag_system(google_api_key=api_key)
    claim = "We achieved 100% carbon neutrality."
    print(f"Testing RAG on claim: {claim}")
    try:
        result = rag.analyze_claim(claim)
        print("RAG result:")
        print(result)
    except Exception as e:
        import traceback
        print(f"Exception: {e}")
        print(traceback.format_exc()) 