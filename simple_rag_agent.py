#!/usr/bin/env python3
"""
Simplified RAG Agent for ESG Greenwashing Detection

This script implements a RAG system using TF-IDF and traditional ML approaches
instead of sentence transformers to avoid dependency conflicts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import pickle
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import re
warnings.filterwarnings('ignore')

# Set style and random seeds
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

class SimpleDocumentProcessor:
    """Simple document processor using TF-IDF."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> list:
        """Split text into chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_documents(self, documents: list) -> list:
        """
        Process documents into chunks.
        
        Args:
            documents: List of document dictionaries with 'title' and 'content'
            
        Returns:
            List of processed documents with chunks
        """
        processed = []
        
        for doc in documents:
            chunks = self.split_text(doc['content'])
            processed.append({
                'title': doc['title'],
                'chunks': chunks,
                'tags': doc.get('tags', []),
                'original_content': doc['content']
            })
        
        return processed

class SimpleVectorStore:
    """Simple vector store using TF-IDF."""
    
    def __init__(self):
        """Initialize vector store."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.documents = []
        self.metadata = []
        self.tfidf_matrix = None
    
    def add_documents(self, documents: list) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries
        """
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Extract all chunks and metadata
        all_chunks = []
        all_metadata = []
        
        for doc in documents:
            # Check if this is a processed document with chunks or a raw document
            if 'chunks' in doc:
                # This is a processed document
                for i, chunk in enumerate(doc['chunks']):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'title': doc['title'],
                        'tags': doc['tags'],
                        'chunk_id': i,
                        'chunk_size': len(chunk)
                    })
            else:
                # This is a raw document, process it
                processor = SimpleDocumentProcessor()
                processed = processor.process_documents([doc])
                if processed:
                    processed_doc = processed[0]
                    for i, chunk in enumerate(processed_doc['chunks']):
                        all_chunks.append(chunk)
                        all_metadata.append({
                            'title': doc['title'],
                            'tags': doc.get('tags', []),
                            'chunk_id': i,
                            'chunk_size': len(chunk)
                        })
        
        # Fit TF-IDF vectorizer and transform documents
        self.tfidf_matrix = self.vectorizer.fit_transform(all_chunks)
        self.documents = all_chunks
        self.metadata = all_metadata
        
        print(f"Vector store now contains {len(self.documents)} chunks")
    
    def search(self, query: str, k: int = 5) -> list:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if self.tfidf_matrix is None or len(self.documents) == 0:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0:
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(similarities[idx]),
                    'rank': i + 1
                })
        
        return results
    
    def save(self, filepath: str) -> None:
        """Save vector store to disk."""
        if self.tfidf_matrix is not None:
            # Save TF-IDF vectorizer
            with open(f"{filepath}_vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save documents and metadata
            data = {
                'documents': self.documents,
                'metadata': self.metadata
            }
            with open(f"{filepath}_data.json", 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load vector store from disk."""
        try:
            # Load TF-IDF vectorizer
            with open(f"{filepath}_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load documents and metadata
            with open(f"{filepath}_data.json", 'r') as f:
                data = json.load(f)
            
            self.documents = data['documents']
            self.metadata = data['metadata']
            
            # Reconstruct TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.transform(self.documents)
            
            print(f"Vector store loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load vector store: {e}")

class SimpleRAGAnalyzer:
    """Simple RAG analyzer using TF-IDF and keyword analysis."""
    
    def __init__(self, vector_store):
        """
        Initialize RAG analyzer.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        
        # Define ESG-related keywords for analysis
        self.esg_keywords = {
            'environmental': [
                'carbon', 'emissions', 'renewable', 'sustainable', 'green', 'eco-friendly',
                'climate', 'energy', 'waste', 'recycling', 'biodiversity', 'pollution'
            ],
            'social': [
                'diversity', 'inclusion', 'equality', 'human rights', 'labor', 'community',
                'health', 'safety', 'education', 'training', 'wellbeing', 'fair trade'
            ],
            'governance': [
                'transparency', 'accountability', 'ethics', 'compliance', 'board',
                'leadership', 'risk management', 'corruption', 'bribery', 'stakeholder'
            ],
            'greenwashing_indicators': [
                'vague', 'unsubstantiated', 'misleading', 'exaggerated', 'cherry-picking',
                'hidden trade-offs', 'false labels', 'irrelevant claims', 'lesser of two evils'
            ]
        }
    
    def analyze_claim(self, claim: str, k: int = 5) -> dict:
        """
        Analyze an ESG claim using RAG.
        
        Args:
            claim: ESG claim text
            k: Number of similar documents to retrieve
            
        Returns:
            Analysis results
        """
        # Search for similar documents
        search_results = self.vector_store.search(claim, k=k)
        
        # Perform keyword analysis
        keyword_analysis = self._analyze_keywords(claim)
        
        # Calculate risk score based on search results and keywords
        risk_score = self._calculate_risk_score(claim, search_results, keyword_analysis)
        
        # Generate analysis
        analysis = {
            'risk_score': risk_score,
            'compliance_score': 1.0 - risk_score,
            'key_issues': keyword_analysis['issues'],
            'relevant_standards': [result['metadata']['title'] for result in search_results[:3]],
            'search_results': search_results,
            'keyword_analysis': keyword_analysis,
            'recommendations': self._generate_recommendations(risk_score, keyword_analysis)
        }
        
        return analysis
    
    def _analyze_keywords(self, claim: str) -> dict:
        """Analyze claim for ESG keywords and potential issues."""
        claim_lower = claim.lower()
        
        # Count keyword matches
        keyword_counts = {}
        for category, keywords in self.esg_keywords.items():
            count = sum(1 for keyword in keywords if keyword in claim_lower)
            keyword_counts[category] = count
        
        # Identify potential issues
        issues = []
        
        # Check for greenwashing indicators
        greenwashing_count = keyword_counts.get('greenwashing_indicators', 0)
        if greenwashing_count > 0:
            issues.append(f"Contains {greenwashing_count} potential greenwashing indicators")
        
        # Check for vague language
        vague_words = ['sustainable', 'green', 'eco-friendly', 'environmentally friendly']
        vague_count = sum(1 for word in vague_words if word in claim_lower)
        if vague_count > 2:
            issues.append("Contains multiple vague environmental terms")
        
        # Check for lack of specificity
        if not any(char.isdigit() for char in claim):
            issues.append("No specific metrics or targets mentioned")
        
        # Check for superlatives
        superlatives = ['best', 'most', 'greatest', 'highest', 'lowest']
        superlative_count = sum(1 for word in superlatives if word in claim_lower)
        if superlative_count > 1:
            issues.append("Contains multiple superlatives without evidence")
        
        return {
            'keyword_counts': keyword_counts,
            'issues': issues,
            'total_esg_keywords': sum(keyword_counts.get(cat, 0) for cat in ['environmental', 'social', 'governance'])
        }
    
    def _calculate_risk_score(self, claim: str, search_results: list, keyword_analysis: dict) -> float:
        """Calculate risk score based on various factors."""
        risk_factors = []
        
        # Factor 1: Search result relevance (lower similarity = higher risk)
        if search_results:
            avg_similarity = np.mean([result['score'] for result in search_results])
            risk_factors.append(1.0 - avg_similarity)
        else:
            risk_factors.append(0.8)  # High risk if no similar documents found
        
        # Factor 2: Greenwashing indicators
        greenwashing_count = keyword_analysis['keyword_counts'].get('greenwashing_indicators', 0)
        risk_factors.append(min(greenwashing_count * 0.2, 1.0))
        
        # Factor 3: Number of issues identified
        issues_count = len(keyword_analysis['issues'])
        risk_factors.append(min(issues_count * 0.15, 1.0))
        
        # Factor 4: Claim length (very short claims might be vague)
        if len(claim) < 50:
            risk_factors.append(0.3)
        elif len(claim) > 500:
            risk_factors.append(0.1)  # Very long claims might be more detailed
        
        # Factor 5: Presence of specific metrics
        has_numbers = any(char.isdigit() for char in claim)
        if not has_numbers:
            risk_factors.append(0.2)
        
        # Calculate weighted average
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Weights for each factor
        weighted_risk = sum(factor * weight for factor, weight in zip(risk_factors, weights))
        
        return min(weighted_risk, 1.0)
    
    def _generate_recommendations(self, risk_score: float, keyword_analysis: dict) -> list:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.append("High risk: Consider providing specific metrics and evidence")
            recommendations.append("Review for potential greenwashing indicators")
        elif risk_score > 0.5:
            recommendations.append("Medium risk: Add more specific details and targets")
            recommendations.append("Consider third-party verification")
        else:
            recommendations.append("Low risk: Claim appears well-supported")
        
        if keyword_analysis['total_esg_keywords'] < 2:
            recommendations.append("Consider incorporating more ESG-specific terminology")
        
        if not any(char.isdigit() for char in keyword_analysis.get('claim', '')):
            recommendations.append("Add specific metrics, targets, or timelines")
        
        return recommendations

def create_esg_corpora():
    """Create ESG corpora for RAG system."""
    print("Creating ESG corpora...")
    
    esg_corpora = {
        'esg_standards': [
            {
                'title': 'GRI Standards - Environmental',
                'content': 'The Global Reporting Initiative (GRI) provides comprehensive standards for environmental reporting. Organizations should report on their environmental impacts including emissions, energy use, water consumption, and biodiversity. Claims must be supported by verifiable data and should include specific targets and timelines.',
                'tags': ['environmental', 'reporting', 'standards', 'GRI']
            },
            {
                'title': 'SASB Standards - Social',
                'content': 'The Sustainability Accounting Standards Board (SASB) focuses on financially material sustainability information. Social topics include labor practices, human rights, and community relations. Companies must provide evidence-based claims and avoid vague or unsubstantiated statements.',
                'tags': ['social', 'accounting', 'standards', 'SASB']
            },
            {
                'title': 'TCFD Framework - Governance',
                'content': 'The Task Force on Climate-related Financial Disclosures (TCFD) provides a framework for climate-related financial risk disclosures. Governance aspects include board oversight, risk management processes, and strategic planning. Claims should be transparent and aligned with business strategy.',
                'tags': ['governance', 'climate', 'risk', 'TCFD']
            },
            {
                'title': 'Greenwashing Prevention Guidelines',
                'content': 'To prevent greenwashing, organizations should avoid vague environmental claims, provide specific evidence, use clear and accurate language, avoid cherry-picking data, and ensure claims are relevant to the product or service. All claims should be verifiable and substantiated.',
                'tags': ['greenwashing', 'prevention', 'guidelines', 'compliance']
            },
            {
                'title': 'ESG Reporting Best Practices',
                'content': 'Effective ESG reporting requires transparency, accuracy, completeness, and consistency. Reports should include clear metrics, targets, and progress indicators. Organizations should avoid misleading statements and ensure all claims are supported by reliable data sources.',
                'tags': ['reporting', 'best practices', 'transparency', 'accuracy']
            }
        ],
        'regulatory_guidelines': [
            {
                'title': 'FTC Green Guides',
                'content': 'The Federal Trade Commission Green Guides help marketers avoid making environmental claims that mislead consumers. Claims must be truthful, not misleading, and substantiated. Vague terms like "eco-friendly" should be avoided unless clearly defined and supported.',
                'tags': ['regulatory', 'FTC', 'marketing', 'claims']
            },
            {
                'title': 'EU Taxonomy Regulation',
                'content': 'The EU Taxonomy Regulation establishes a classification system for environmentally sustainable economic activities. Claims about environmental sustainability must align with specific technical screening criteria and avoid misleading consumers about environmental benefits.',
                'tags': ['EU', 'taxonomy', 'regulation', 'sustainability']
            }
        ]
    }
    
    return esg_corpora

def main():
    """Main execution function."""
    print("🚀 SIMPLIFIED RAG AGENT FOR ESG GREENWASHING DETECTION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Load and prepare data
        print("=== DATA LOADING AND PREPARATION ===\n")
        df = pd.read_parquet('data/clean_claims.parquet')
        print(f"Dataset shape: {df.shape}")
        print(f"Greenwashing rate: {df['greenwashing_flag'].mean():.2%}")
        
        # Remove rows with missing targets
        df_model = df.dropna(subset=['greenwashing_flag', 'esg_claim_text'])
        print(f"Rows after removing missing targets: {len(df_model)}")
        
        # Step 2: Set up RAG system
        print("\n=== RAG SYSTEM SETUP ===\n")
        
        # Create directories
        os.makedirs('data/corpora', exist_ok=True)
        os.makedirs('data/vector_stores', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('metrics', exist_ok=True)
        
        # Create ESG corpora
        print("Creating ESG corpora...")
        esg_corpora = create_esg_corpora()
        
        print(f"Created {len(esg_corpora)} ESG corpora:")
        for corpus_name, corpus_data in esg_corpora.items():
            print(f"  - {corpus_name}: {len(corpus_data)} documents")
        
        # Save corpora
        for corpus_name, corpus_data in esg_corpora.items():
            corpus_file = f'data/corpora/{corpus_name}.json'
            with open(corpus_file, 'w') as f:
                json.dump(corpus_data, f, indent=2)
            print(f"Saved: {corpus_file}")
        
        # Step 3: Process documents
        print("\n=== DOCUMENT PROCESSING ===\n")
        
        # Initialize document processor
        processor = SimpleDocumentProcessor()
        print("Document processor initialized")
        
        # Process ESG corpora
        processed_docs = {}
        for corpus_name, corpus_data in esg_corpora.items():
            print(f"\nProcessing {corpus_name}...")
            processed = processor.process_documents(corpus_data)
            processed_docs[corpus_name] = processed
            print(f"  Processed {len(processed)} documents")
            
            # Show sample processed document
            if processed:
                sample_doc = processed[0]
                print(f"  Sample processed document:")
                print(f"    Title: {sample_doc['title']}")
                print(f"    Chunks: {len(sample_doc['chunks'])}")
                print(f"    First chunk: {sample_doc['chunks'][0][:100]}...")
        
        # Save processed documents
        for corpus_name, processed in processed_docs.items():
            processed_file = f'data/corpora/{corpus_name}_processed.json'
            with open(processed_file, 'w') as f:
                json.dump(processed, f, indent=2)
            print(f"Saved: {processed_file}")
        
        # Step 4: Set up vector store
        print("\n=== VECTOR STORE SETUP ===\n")
        
        # Initialize vector store
        vector_store = SimpleVectorStore()
        print("Vector store initialized")
        
        # Add documents to vector store
        total_docs = 0
        for corpus_name, processed in processed_docs.items():
            print(f"\nAdding {corpus_name} to vector store...")
            
            # Add processed documents directly to vector store
            vector_store.add_documents(processed)
            total_docs += sum(len(doc['chunks']) for doc in processed)
            print(f"  Added {sum(len(doc['chunks']) for doc in processed)} chunks")
        
        print(f"\nTotal chunks in vector store: {total_docs}")
        
        # Save vector store
        vector_store.save('data/vector_stores/esg_vector_store')
        print("Vector store saved to data/vector_stores/esg_vector_store")
        
        # Step 5: Set up RAG analyzer
        print("\n=== RAG ANALYZER SETUP ===\n")
        
        # Initialize RAG analyzer
        rag_analyzer = SimpleRAGAnalyzer(vector_store)
        print("RAG analyzer initialized")
        
        # Step 6: Test RAG analyzer
        print("\n=== RAG ANALYZER TESTING ===\n")
        
        # Test RAG analyzer with sample claims
        print("Testing RAG analyzer with sample claims...")
        
        # Get sample claims from dataset
        greenwashing_claims = df_model[df_model['greenwashing_flag'] == 1]['esg_claim_text'].tolist()
        legitimate_claims = df_model[df_model['greenwashing_flag'] == 0]['esg_claim_text'].tolist()
        
        sample_claims = greenwashing_claims[:3] + legitimate_claims[:2]
        
        for i, claim in enumerate(sample_claims):
            print(f"\nTest Claim {i+1}: {str(claim)[:100]}...")
            
            # Analyze claim
            analysis = rag_analyzer.analyze_claim(str(claim))
            
            print(f"  Risk Score: {analysis['risk_score']:.2f}")
            print(f"  Compliance Score: {analysis['compliance_score']:.2f}")
            print(f"  Key Issues: {analysis['key_issues'][:3]}")
            print(f"  Relevant Standards: {analysis['relevant_standards'][:2]}")
            print(f"  Recommendations: {analysis['recommendations'][:1]}")
        
        # Step 7: Evaluate RAG performance
        print("\n=== RAG PERFORMANCE EVALUATION ===\n")
        
        # Create test set
        test_claims = df_model.sample(n=min(50, len(df_model)), random_state=42)
        print(f"Test set size: {len(test_claims)}")
        
        # Analyze test claims
        rag_results = []
        for idx, row in test_claims.iterrows():
            claim = str(row['esg_claim_text'])
            actual_greenwashing = row['greenwashing_flag']
            
            # Get RAG analysis
            analysis = rag_analyzer.analyze_claim(claim)
            
            # Determine prediction based on risk score
            predicted_greenwashing = 1 if analysis['risk_score'] > 0.5 else 0
            
            rag_results.append({
                'claim': claim,
                'actual': actual_greenwashing,
                'predicted': predicted_greenwashing,
                'risk_score': analysis['risk_score'],
                'compliance_score': analysis['compliance_score'],
                'key_issues': analysis['key_issues'],
                'relevant_standards': analysis['relevant_standards']
            })
        
        # Calculate metrics
        actual = [r['actual'] for r in rag_results]
        predicted = [r['predicted'] for r in rag_results]
        risk_scores = [r['risk_score'] for r in rag_results]
        
        rag_metrics = {
            'accuracy': accuracy_score(actual, predicted),
            'precision': precision_score(actual, predicted),
            'recall': recall_score(actual, predicted),
            'f1_score': f1_score(actual, predicted),
            'roc_auc': roc_auc_score(actual, risk_scores)
        }
        
        print("RAG Performance Metrics:")
        for metric, value in rag_metrics.items():
            print(f"  {metric.capitalize()}: {value:.3f}")
        
        # Step 8: Save RAG system
        print("\n=== SAVING RAG SYSTEM ===\n")
        
        # Save RAG analyzer
        with open('models/rag_analyzer.pkl', 'wb') as f:
            pickle.dump(rag_analyzer, f)
        print("Saved: rag_analyzer.pkl")
        
        # Save RAG metrics
        print("\n=== SAVING RAG METRICS ===\n")
        
        rag_system_metrics = {
            'rag_performance': rag_metrics,
            'system_info': {
                'total_corpora': len(esg_corpora),
                'total_documents': sum(len(corpus) for corpus in esg_corpora.values()),
                'total_chunks': total_docs,
                'test_set_size': len(test_claims)
            },
            'corpora_info': {
                name: {
                    'documents': len(corpus),
                    'processed_chunks': len(processed_docs[name])
                } for name, corpus in esg_corpora.items()
            }
        }
        
        with open('metrics/rag_system_metrics.json', 'w') as f:
            json.dump(rag_system_metrics, f, indent=2)
        print("Saved: rag_system_metrics.json")
        
        # Step 9: Print summary
        print("\n=== RAG AGENT SUMMARY ===\n")
        
        print("1. ESG CORPORA:")
        print(f"   - Total corpora: {len(esg_corpora)}")
        print(f"   - Total documents: {sum(len(corpus) for corpus in esg_corpora.values())}")
        print(f"   - Total chunks: {total_docs}")
        
        print("\n2. RAG PERFORMANCE:")
        for metric, value in rag_metrics.items():
            print(f"   {metric.capitalize()}: {value:.3f}")
        
        print("\n3. KEY INSIGHTS:")
        print("   - RAG system provides interpretable results with regulatory context")
        print("   - TF-IDF approach avoids dependency conflicts")
        print("   - System can identify specific compliance issues")
        print("   - Provides actionable recommendations")
        
        print("\n4. NEXT STEPS:")
        print("   - RAG system ready for deployment")
        print("   - Can be integrated into Streamlit app")
        print("   - Consider expanding ESG corpora with more recent regulations")
        print("   - Proceed to business plan generation")
        
        print(f"\n✅ RAG agent completed successfully at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error during RAG agent execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 