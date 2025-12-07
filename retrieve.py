"""
retrieve.py - Vector Store logic with relaxed matching
"""
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os

class VectorStore:
    def __init__(self):
        os.makedirs("./chroma_db", exist_ok=True)
        
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="places")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_place(self, place: Dict):
        """Add a place to the vector store"""
        text = f"{place['name']} {place.get('description', '')} {place.get('address', '')} {','.join(place.get('types', []))}"
        embedding = self.model.encode(text).tolist()
        
        metadata = {
            "name": place['name'],
            "rating": str(place.get('rating', 0)),
            "description": place.get('description', '')[:200], # Truncate for safety
            "source": place.get('source', 'unknown'),
            "vibe_tags": ",".join(place.get('attributes', {}).get('vibe', []))
        }
        
        doc_id = str(place.get('place_id', hash(text)))
        
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

    def hybrid_search(self, query: str, user_context: Dict, n_results: int = 5) -> List[Dict]:
        """
        Semantic search + Metadata filtering
        """
        query_embedding = self.model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 3  
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []

        processed_results = []
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] 
        documents = results['documents'][0]

        for i in range(len(ids)):
            if distances[i] > 1.0: 
                continue
                
            processed_results.append({
                'id': ids[i],
                'metadata': metadatas[i],
                'document': documents[i],
                'distance': distances[i],
                'score': 2.0 - distances[i] 
            })

        target_vibe = 'adventurous' if user_context.get('user_type') == 'Trekker' else None
        
        if target_vibe:
            for res in processed_results:
                if target_vibe in res['metadata'].get('vibe_tags', ''):
                    res['score'] += 0.5 

        processed_results.sort(key=lambda x: x['score'], reverse=True)
        
        return processed_results[:n_results]
