from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model, documents):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Parameters:
    - model: The LLM model used for processing.
    - documents: List of documents to be embedded.

    Returns:
    - query_engine: Query engine for retrieving relevant document chunks.
    """
    try:
        logging.info("Initializing Gemini Embedding model...")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        
        logging.info("Configuring LLM settings...")
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20
        
        logging.info("Creating Vector Store Index from documents...")
        index = VectorStoreIndex.from_documents(documents)
        
        logging.info("Persisting index storage...")
        index.storage_context.persist()
        
        logging.info("Creating Query Engine...")
        query_engine = index.as_query_engine()
        
        return query_engine
    except Exception as e:
        logging.error(f"Error in download_gemini_embedding: {str(e)}")
        raise customexception(e, sys)
