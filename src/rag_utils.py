"""
RAG utilities for ESG fraud detection.

This module provides functions for embedding, vector database operations,
retrieval, and prompt templates for the RAG-based fraud detection system.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import re

# RAG and embedding imports
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class ESGDocumentProcessor:
    """Process and chunk ESG documents for RAG."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Process text into chunks.
        
        Args:
            text: Input text
            metadata: Document metadata
            
        Returns:
            List of Document chunks
        """
        if metadata is None:
            metadata = {}
        
        chunks = self.text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['chunk_size'] = len(chunk)
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        return documents

class ESGVectorStore:
    """Vector store for ESG documents using FAISS."""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', dimension: int = 384):
        """
        Initialize vector store.
        
        Args:
            embedding_model: Name of the sentence transformer model
            dimension: Embedding dimension
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.metadata = []
        
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
        """
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Initialize FAISS index if not exists
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(metadata)
        
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with documents and scores
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def save(self, filepath: str) -> None:
        """Save vector store to disk."""
        if self.index is not None:
            faiss.write_index(self.index, f"{filepath}.index")
            
            # Save documents and metadata
            data = {
                'documents': self.documents,
                'metadata': self.metadata
            }
            with open(f"{filepath}.json", 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load vector store from disk."""
        try:
            self.index = faiss.read_index(f"{filepath}.index")
            
            with open(f"{filepath}.json", 'r') as f:
                data = json.load(f)
            
            self.documents = data['documents']
            self.metadata = data['metadata']
            
            logger.info(f"Vector store loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")

class ChromaVectorStore:
    """Alternative vector store using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist the database
        """
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        self.collection = None
        
    def create_collection(self, name: str = "esg_documents") -> None:
        """Create a new collection."""
        try:
            self.collection = self.client.get_collection(name=name)
            logger.info(f"Using existing collection: {name}")
        except:
            self.collection = self.client.create_collection(name=name)
            logger.info(f"Created new collection: {name}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to ChromaDB."""
        if self.collection is None:
            self.create_collection()
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search documents in ChromaDB."""
        if self.collection is None:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i],
                'rank': i + 1
            })
        
        return search_results

class ESGRAgAnalyzer:
    """RAG-based ESG claim analyzer."""
    
    def __init__(self, vector_store, openai_api_key: str = None):
        """
        Initialize RAG analyzer.
        
        Args:
            vector_store: Vector store instance
            openai_api_key: OpenAI API key
        """
        self.vector_store = vector_store
        
        # Initialize OpenAI if API key provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, convert_system_message_to_human=True)
        else:
            self.llm = None
        
        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "claim"],
            template="""
[SYSTEM] You are a forensic sustainability analyst specializing in ESG fraud detection and greenwashing identification. Your role is to analyze ESG claims against regulatory frameworks and industry standards to identify potential misrepresentations or non-compliance.

[CONTEXT] The following are relevant regulatory documents, standards, and guidelines:
{context}

[CLAIM] ESG Claim to analyze: "{claim}"

Based on the context provided, please answer the following questions:

Question 1: Which specific KPI(s) or metrics does this claim refer to?
Question 2: Does the retrieved regulation, standard, or guideline contradict or cast doubt on this claim?
Question 3: What is the risk level (0-1) of this claim being misleading or non-compliant?

Please provide your analysis in the following JSON format:
{{
    "kpi": "specific KPI or metric mentioned",
    "risk_score": 0.0-1.0,
    "rationale": "detailed explanation of your assessment",
    "regulatory_issues": ["list of any regulatory concerns"],
    "recommendations": ["suggestions for improvement or verification"]
}}

Focus on factual analysis based on the provided context. If the context doesn't contain relevant information, indicate this in your rationale.
"""
        )
    
    def analyze_claim(self, claim: str, k: int = 5) -> Dict[str, Any]:
        """
        Analyze an ESG claim using RAG.
        
        Args:
            claim: ESG claim text
            k: Number of relevant documents to retrieve
            
        Returns:
            Analysis results
        """
        # Retrieve relevant documents
        search_results = self.vector_store.search(claim, k=k)
        
        if not search_results:
            return {
                "error": "No relevant documents found",
                "risk_score": 0.5,
                "rationale": "Unable to analyze due to lack of relevant context"
            }
        
        # Prepare context from retrieved documents
        context_parts = []
        for result in search_results:
            context_parts.append(f"Document {result['rank']} (Score: {result['score']:.3f}):\n{result['document']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate analysis using LLM if available
        if self.llm:
            try:
                prompt = self.prompt_template.format(context=context, claim=claim)
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                # Try to parse JSON response
                try:
                    analysis = json.loads(response.content)
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    analysis = {
                        "kpi": "Unable to extract",
                        "risk_score": 0.5,
                        "rationale": response.content,
                        "regulatory_issues": [],
                        "recommendations": []
                    }
                
                # Add metadata
                analysis["retrieved_documents"] = len(search_results)
                analysis["top_document_score"] = search_results[0]["score"] if search_results else 0
                
                return analysis
                
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                return {
                    "error": f"LLM analysis failed: {str(e)}",
                    "risk_score": 0.5,
                    "rationale": "Analysis failed due to technical error"
                }
        else:
            # Fallback analysis without LLM
            return self._fallback_analysis(claim, search_results)
    
    def _fallback_analysis(self, claim: str, search_results: List[Dict]) -> Dict[str, Any]:
        """Fallback analysis when LLM is not available."""
        # Simple keyword-based analysis
        risk_keywords = ['vague', 'unclear', 'unspecified', 'general', 'commitment', 'will', 'plan']
        high_risk_keywords = ['guarantee', 'promise', 'ensure', 'definitely', 'certainly']
        
        claim_lower = claim.lower()
        risk_score = 0.0
        
        # Check for high-risk keywords
        for keyword in high_risk_keywords:
            if keyword in claim_lower:
                risk_score += 0.3
        
        # Check for vague language
        for keyword in risk_keywords:
            if keyword in claim_lower:
                risk_score += 0.1
        
        # Normalize risk score
        risk_score = min(1.0, risk_score)
        
        return {
            "kpi": "Keyword-based extraction",
            "risk_score": risk_score,
            "rationale": f"Fallback analysis based on keyword detection. Risk score: {risk_score:.2f}",
            "regulatory_issues": [],
            "recommendations": ["Use LLM for more detailed analysis"],
            "retrieved_documents": len(search_results),
            "top_document_score": search_results[0]["score"] if search_results else 0
        }

class ESG_RAG_System:
    """A RAG system for ESG claim analysis."""
    def __init__(self, rag_chain, vector_store):
        self.rag_chain = rag_chain
        self.vector_store = vector_store

    def analyze_claim(self, claim_text: str) -> Dict[str, Any]:
        """Analyzes a claim using the RAG chain."""
        if not self.rag_chain:
            # When no LLM is available, use keyword-based fallback analysis
            return self._fallback_keyword_analysis(claim_text)
        
        try:
            result = self.rag_chain({"query": claim_text})
            
            # Improved risk scoring logic
            source_docs = result.get("source_documents", [])
            analysis_text = result.get("result", "")
            
            # Calculate risk based on multiple factors
            risk_score = self._calculate_rag_risk(claim_text, source_docs, analysis_text)
            
            return {
                "rag_risk_score": risk_score,
                "analysis": analysis_text,
                "sources": [doc.metadata.get('source', 'Unknown') for doc in source_docs]
            }
        except Exception as e:
            logger.error(f"RAG analysis failed: {e}")
            return self._fallback_keyword_analysis(claim_text)
    
    def _fallback_keyword_analysis(self, claim_text: str) -> Dict[str, Any]:
        """Fallback analysis when LLM is not available."""
        # Enhanced keyword-based risk analysis
        risk_keywords = [
            'vague', 'unclear', 'unspecified', 'general', 'commitment', 
            'will', 'plan', 'aim', 'target', 'goal', 'strive', 'endeavor'
        ]
        high_risk_keywords = [
            'guarantee', 'promise', 'ensure', 'definitely', 'certainly',
            '100%', 'fully', 'completely', 'absolutely', 'never', 'always'
        ]
        extreme_claims = [
            'carbon neutral', 'net zero', 'zero emissions', '100% renewable',
            'fully sustainable', 'completely green'
        ]
        
        claim_lower = claim_text.lower()
        risk_score = 0.0
        
        # Check for extreme claims (highest risk)
        for keyword in extreme_claims:
            if keyword in claim_lower:
                risk_score += 0.4
                break  # Only count once for extreme claims
        
        # Check for high-risk keywords
        for keyword in high_risk_keywords:
            if keyword in claim_lower:
                risk_score += 0.2
        
        # Check for vague language
        for keyword in risk_keywords:
            if keyword in claim_lower:
                risk_score += 0.1
        
        # Normalize risk score
        risk_score = min(1.0, risk_score)
        
        return {
            "rag_risk_score": risk_score,
            "analysis": f"Keyword-based analysis: Risk score {risk_score:.2f}. " + 
                       ("High risk indicators detected." if risk_score > 0.5 else 
                        "Moderate risk indicators detected." if risk_score > 0.2 else 
                        "Low risk indicators detected."),
            "sources": ["Keyword analysis (no LLM available)"]
        }
    
    def _calculate_rag_risk(self, claim_text: str, source_docs: list, analysis_text: str) -> float:
        """Calculate RAG risk score based on multiple factors."""
        risk_score = 0.0
        
        # Factor 1: Number of relevant sources found (more sources = lower risk)
        if source_docs:
            # If sources are found, base risk on source relevance
            avg_source_score = sum(doc.metadata.get('score', 0.5) for doc in source_docs) / len(source_docs)
            risk_score += (1.0 - avg_source_score) * 0.3  # Higher relevance = lower risk
        else:
            # No sources found = higher risk (claim not supported by regulations)
            risk_score += 0.6
        
        # Factor 2: Analysis sentiment (negative analysis = higher risk)
        negative_indicators = ['vague', 'unclear', 'unsupported', 'risky', 'concerning', 'problematic']
        positive_indicators = ['clear', 'supported', 'compliant', 'transparent', 'verified']
        
        analysis_lower = analysis_text.lower()
        negative_count = sum(1 for indicator in negative_indicators if indicator in analysis_lower)
        positive_count = sum(1 for indicator in positive_indicators if indicator in analysis_lower)
        
        if negative_count > positive_count:
            risk_score += 0.3
        elif positive_count > negative_count:
            risk_score -= 0.2  # Reduce risk for positive indicators
        
        # Factor 3: Claim characteristics
        claim_lower = claim_text.lower()
        if any(extreme in claim_lower for extreme in ['100%', 'fully', 'completely', 'guarantee']):
            risk_score += 0.2
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, risk_score))

def download_esg_corpora(docs_path: str = "business") -> List[Document]:
    """
    Download and process ESG regulatory documents.
    
    Returns:
        List of processed documents
    """
    logger.info("Downloading ESG regulatory documents...")
    
    # Sample ESG documents (in practice, you'd download real PDFs/HTMLs)
    sample_docs = [
        {
            "title": "EU Taxonomy Regulation",
            "content": """
            The EU Taxonomy Regulation establishes a classification system for environmentally sustainable economic activities.
            To be considered environmentally sustainable, an economic activity must:
            1. Contribute substantially to one or more of the six environmental objectives
            2. Do no significant harm to any of the other environmental objectives
            3. Be carried out in compliance with minimum social safeguards
            4. Comply with technical screening criteria
            
            Environmental objectives include:
            - Climate change mitigation
            - Climate change adaptation
            - Sustainable use and protection of water and marine resources
            - Transition to a circular economy
            - Pollution prevention and control
            - Protection and restoration of biodiversity and ecosystems
            """,
            "source": "EU Regulation 2020/852"
        },
        {
            "title": "GRI Standards - Climate Change",
            "content": """
            GRI 305: Emissions 2016 sets out reporting requirements on emissions for organizations.
            Organizations should report:
            - Direct (Scope 1) GHG emissions
            - Energy indirect (Scope 2) GHG emissions
            - Other indirect (Scope 3) GHG emissions
            - Emissions intensity
            - Reduction of emissions
            
            Claims about emissions reductions must be:
            - Quantified and verifiable
            - Based on consistent methodologies
            - Supported by evidence
            - Disclosed transparently
            """,
            "source": "GRI Standards"
        },
        {
            "title": "Transition Finance Guidelines",
            "content": """
            Transition finance supports companies in high-emitting sectors to reduce their environmental impact.
            Key principles:
            1. Credible transition plans with clear targets
            2. Science-based pathways aligned with Paris Agreement
            3. Regular reporting and verification
            4. No significant harm to other environmental objectives
            
            Red flags for greenwashing:
            - Vague or unquantified commitments
            - Lack of interim targets
            - No verification mechanisms
            - Inconsistent with science-based pathways
            """,
            "source": "Transition Finance Framework"
        }
    ]
    
    # Process documents
    processor = ESGDocumentProcessor()
    all_documents = []
    
    for doc in sample_docs:
        metadata = {
            "title": doc["title"],
            "source": doc["source"],
            "date_added": datetime.now().isoformat()
        }
        
        documents = processor.process_text(doc["content"], metadata)
        all_documents.extend(documents)
    
    logger.info(f"Processed {len(all_documents)} document chunks")
    return all_documents

def create_rag_system(
    docs_path: str = "business", 
    use_chroma: bool = False,
    google_api_key: Optional[str] = None
) -> "ESG_RAG_System":
    """
    Create a complete RAG system for ESG analysis using Google Gemini.
    """
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key

    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not found. RAG will not use an LLM.")
        llm = None
        # Use local sentence transformer for embeddings when no API key
        from sentence_transformers import SentenceTransformer
        embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, convert_system_message_to_human=True)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    documents = download_esg_corpora(docs_path)
    
    vector_store_path = "models/rag_system/vector_store.index"
    if use_chroma:
        import chromadb
        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    else:
        if os.path.exists(vector_store_path):
             vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        else:
             vector_store = FAISS.from_documents(documents, embeddings)
             vector_store.save_local(vector_store_path)

    rag_chain = None
    if llm:
        prompt_template = """
        Analyze the following ESG claim based on the provided context from regulatory documents. 
        Assess the risk of greenwashing on a scale of 0 to 1.
        Explain your reasoning and cite specific sources from the context.

        CONTEXT: {context}
        CLAIM: {question}
        
        ANALYSIS:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    return ESG_RAG_System(rag_chain, vector_store) 