# retrieve.py
import os
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self):
        os.makedirs("./simple_store", exist_ok=True)
        self.vectorizer = TfidfVectorizer(max_features=4000, stop_words='english')
        self.documents = []   
        self.metadatas = []   
        self.ids = []         
        self._fitted = False
        self._matrix = None

    def _ensure_vectorizer(self):
        if (not self._fitted) and self.documents:
            self._matrix = self.vectorizer.fit_transform(self.documents).toarray()
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True) + 1e-12
            self._matrix = self._matrix / norms
            self._fitted = True

    def add_place(self, place: Dict):
        """Add or update a place. `place` should include 'name' and 'description' or 'address'."""
        text = f"{place.get('name','')} {place.get('description','')} {place.get('address','')}"
        doc_id = str(place.get('place_id', hash(text)))
        if doc_id in self.ids:
            idx = self.ids.index(doc_id)
            self.documents[idx] = text
            self.metadatas[idx] = {
                "name": place.get('name'),
                "rating": str(place.get('rating', 0)),
                "description": (place.get('description') or "")[:200],
                "source": place.get('source', 'unknown'),
                "vibe_tags": ",".join(place.get('attributes', {}).get('vibe', []))
            }
        else:
            self.ids.append(doc_id)
            self.documents.append(text)
            self.metadatas.append({
                "name": place.get('name'),
                "rating": str(place.get('rating', 0)),
                "description": (place.get('description') or "")[:200],
                "source": place.get('source', 'unknown'),
                "vibe_tags": ",".join(place.get('attributes', {}).get('vibe', []))
            })
        self._fitted = False
        self._matrix = None

    def hybrid_search(self, query: str, user_context: Dict, n_results: int = 5) -> List[Dict]:
        """
        Perform TF-IDF -> cosine similarity search.
        Returns list of dicts with keys: id, metadata, document, distance, score
        """
        if not self.documents:
            return []

        self._ensure_vectorizer()

        q_vec = self.vectorizer.transform([query]).toarray()
        q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)

        sims = cosine_similarity(q_vec, self._matrix)[0]  # similarity in [0,1]
        distances = 1.0 - sims

        results = []
        for idx in np.argsort(-sims)[: min(len(sims), n_results * 3)]:
            if distances[idx] > 1.0:
                continue
            results.append({
                'id': self.ids[idx],
                'metadata': self.metadatas[idx],
                'document': self.documents[idx],
                'distance': float(distances[idx]),
                'score': float(sims[idx])
            })

        target_vibe = 'adventurous' if user_context.get('user_type') == 'Trekker' else None
        if target_vibe:
            for r in results:
                if target_vibe in (r['metadata'].get('vibe_tags') or ''):
                    r['score'] += 0.2

        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:n_results]
