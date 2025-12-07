# attribute_classifier.py
"""
Attribute Classifier - Lightweight TF-IDF + DBSCAN + heuristic fallback
Replaces heavy sentence-transformers / ctransformers usage.
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import json
import os
import pickle

VALID_TAGS = ['nature', 'shopping', 'spiritual', 'foodie', 'adventure', 'history', 'family', 'luxury', 'budget', 'general']

class HeuristicLLM:
    def __init__(self):
        self.valid_tags = VALID_TAGS

    def generate_tags(self, text_samples: List[str]) -> List[str]:
        text = " ".join(text_samples).lower()
        tags = set()
        if any(w in text for w in ('mall', 'shop', 'boutique')): tags.add('shopping')
        if any(w in text for w in ('waterfall', 'park', 'forest', 'nature', 'river')): tags.add('nature')
        if any(w in text for w in ('temple', 'church', 'mosque', 'shrine')): tags.add('spiritual')
        if any(w in text for w in ('cafe', 'food', 'restaurant', 'eat')): tags.add('foodie')
        if any(w in text for w in ('trek', 'hike', 'trail', 'adventure', 'climb')): tags.add('adventure')
        if any(w in text for w in ('museum', 'fort', 'monument', 'history')): tags.add('history')
        return list(tags) if tags else ['general']


class AttributeClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.cluster_model = None
        self.knn = None
        self.llm = HeuristicLLM()

        self.cluster_labels_map = {}
        self.embeddings_matrix = None
        self.ids = []
        self.is_fitted = False
        self.eps = 0.5  # DBSCAN cosine threshold

    def fit_clusters(self, places_data: List[Dict]):
        """
        Fit TF-IDF on existing places and cluster using DBSCAN.
        places_data: list of dict with 'name' and optional 'description'
        """
        if not places_data:
            return

        texts = [f"{p.get('name','')} {p.get('description','')}" for p in places_data]
        X = self.vectorizer.fit_transform(texts)
        X_dense = X.toarray()
        # Normalize rows for cosine distance to be consistent
        norms = np.linalg.norm(X_dense, axis=1, keepdims=True) + 1e-12
        Xn = X_dense / norms

        self.cluster_model = DBSCAN(eps=self.eps, min_samples=2, metric='cosine')
        clusters = self.cluster_model.fit_predict(Xn)

        unique_clusters = set(clusters)
        self.cluster_labels_map = {}
        for c_id in unique_clusters:
            if c_id == -1:
                continue
            indices = [i for i, c in enumerate(clusters) if c == c_id]
            samples = [texts[i] for i in indices[:5]]
            tags = self.llm.generate_tags(samples)
            self.cluster_labels_map[c_id] = tags

        mask = [i for i, c in enumerate(clusters) if c != -1]
        if mask:
            self.embeddings_matrix = Xn[mask]
            self.ids = [clusters[i] for i in mask]
            self.knn = NearestNeighbors(n_neighbors=1, metric='cosine')
            self.knn.fit(self.embeddings_matrix)
            self.is_fitted = True

    def classify_user_type(self, query: str) -> str:
        q = (query or "").lower()
        if 'trek' in q or 'adventure' in q: return 'Trekker'
        if 'food' in q or 'cafe' in q: return 'Foodie'
        if 'shop' in q or 'mall' in q: return 'Shopper'
        if 'temple' in q or 'church' in q: return 'Pilgrim'
        return "General"

    def classify_with_rules(self, place: Dict) -> Dict:
        """
        Use TF-IDF + KNN to inherit cluster tags if close; otherwise fallback to heuristics.
        """
        text = f"{place.get('name','')} {place.get('description','')}"
        if not self.is_fitted:
            tags = self.llm.generate_tags([text])
            return {"vibe": tags, "duration": ["half_day"], "best_time": ["any"]}

        v = self.vectorizer.transform([text]).toarray()
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

        dist, idx = self.knn.kneighbors(v)
        if dist[0][0] < self.eps:
            cluster_id = self.ids[idx[0][0]]
            tags = self.cluster_labels_map.get(cluster_id, ['general'])
        else:
            tags = self.llm.generate_tags([text])

        return {
            "vibe": tags,
            "duration": ["half_day"],
            "best_time": ["day"]
        }
