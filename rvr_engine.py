"""Core RVR (Retrieve-Verify-Retrieve) implementation"""

from typing import List, Dict, Any, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RVREngine:
    """Implements the Retrieve-Verify-Retrieve algorithm"""
    
    def __init__(self, document_store, model_name: str = 'all-MiniLM-L6-v2'):
        self.doc_store = document_store
        self.encoder = SentenceTransformer(model_name)
        self.verify_threshold = 0.4  # Relevance threshold for verification
        
    def retrieve(self, query: str, top_k: int = 10, excluded_ids: Set[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top-k documents using semantic similarity"""
        query_embedding = self.encoder.encode([query])
        
        # Compute similarities
        similarities = cosine_similarity(
            query_embedding,
            self.doc_store.embeddings
        )[0]
        
        # Exclude already retrieved documents
        if excluded_ids:
            for doc_id in excluded_ids:
                similarities[doc_id] = -1
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Skip excluded
                results.append({
                    'id': int(idx),
                    'document': self.doc_store.documents[idx],
                    'score': float(similarities[idx])
                })
        
        return results
    
    def verify(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verify candidates based on relevance threshold and diversity"""
        verified = []
        
        for candidate in candidates:
            # Simple verification: threshold-based filtering
            if candidate['score'] >= self.verify_threshold:
                verified.append(candidate)
        
        # Additional verification: ensure diversity (simple deduplication)
        verified = self._ensure_diversity(verified)
        
        return verified
    
    def _ensure_diversity(self, documents: List[Dict[str, Any]], similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """Remove near-duplicate documents"""
        if len(documents) <= 1:
            return documents
        
        diverse_docs = [documents[0]]
        doc_texts = [doc['document']['text'] for doc in documents]
        doc_embeddings = self.encoder.encode(doc_texts)
        
        for i in range(1, len(documents)):
            # Check similarity with already selected docs
            is_diverse = True
            for j in range(len(diverse_docs)):
                sim = cosine_similarity(
                    doc_embeddings[i].reshape(1, -1),
                    doc_embeddings[diverse_docs[j]['id']].reshape(1, -1)
                )[0][0]
                
                if sim > similarity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_docs.append(documents[i])
        
        return diverse_docs
    
    def augment_query(self, original_query: str, verified_docs: List[Dict[str, Any]]) -> str:
        """Augment query with information from verified documents"""
        if not verified_docs:
            return original_query
        
        # Extract key terms from verified documents
        verified_texts = [doc['document']['text'][:200] for doc in verified_docs]
        augmentation = " ".join(verified_texts[:2])  # Use first 2 docs
        
        # Create augmented query
        augmented = f"{original_query} [Context: {augmentation[:300]}]"
        return augmented
    
    def search(self, query: str, max_rounds: int = 3, top_k: int = 5) -> Dict[str, Any]:
        """Execute full RVR search pipeline"""
        all_verified = []
        excluded_ids = set()
        rounds_info = []
        
        current_query = query
        
        for round_num in range(1, max_rounds + 1):
            # Retrieve
            candidates = self.retrieve(
                current_query,
                top_k=top_k * 2,  # Retrieve more to account for verification
                excluded_ids=excluded_ids
            )
            
            if not candidates:
                break  # No more documents to retrieve
            
            # Verify
            verified = self.verify(current_query, candidates)
            
            # Track verified documents
            for doc in verified:
                excluded_ids.add(doc['id'])
                all_verified.append(doc)
            
            rounds_info.append({
                'round': round_num,
                'retrieved_count': len(candidates),
                'verified_count': len(verified),
                'verified_documents': verified
            })
            
            # Augment query for next round
            if round_num < max_rounds and verified:
                current_query = self.augment_query(query, verified)
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage(all_verified)
        
        return {
            'query': query,
            'rounds': rounds_info,
            'total_verified': len(all_verified),
            'coverage_metrics': coverage_metrics
        }
    
    def _calculate_coverage(self, verified_docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate coverage and diversity metrics"""
        if not verified_docs:
            return {'diversity_score': 0.0, 'avg_score': 0.0}
        
        scores = [doc['score'] for doc in verified_docs]
        
        # Diversity: measure avg pairwise distance
        if len(verified_docs) > 1:
            embeddings = self.encoder.encode([doc['document']['text'] for doc in verified_docs])
            similarities = cosine_similarity(embeddings)
            # Diversity = 1 - avg similarity (excluding diagonal)
            n = len(similarities)
            total_sim = (similarities.sum() - n) / (n * (n - 1)) if n > 1 else 0
            diversity_score = 1 - total_sim
        else:
            diversity_score = 0.0
        
        return {
            'diversity_score': float(diversity_score),
            'avg_relevance_score': float(np.mean(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores))
        }
