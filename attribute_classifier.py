"""
Attribute Classifier - Hybrid (DBSCAN Clustering + TinyLlama Labeling)
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import json
import os
import pickle

try:
    from ctransformers import AutoModelForCausalLM
except ImportError:
    print("⚠️ ctransformers not installed. Run: pip install ctransformers>=0.2.27")
    AutoModelForCausalLM = None

class TinyLlamaWrapper:
    def __init__(self, model_path="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        self.model = None
        self.valid_tags = ['nature', 'shopping', 'spiritual', 'foodie', 'adventure', 'history', 'family', 'luxury', 'budget']
        
        if AutoModelForCausalLM:
            try:
                print("🦙 Loading TinyLlama...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    model_file=model_file, 
                    model_type="llama", 
                    context_length=1024,
                    gpu_layers=0
                )
                print("✅ TinyLlama loaded successfully.")
            except Exception as e:
                print(f"❌ Failed to load TinyLlama: {e}. Using heuristic fallback.")

    def generate_tags(self, text_samples: List[str]) -> List[str]:
        """
        Uses TinyLlama to summarize a cluster of texts into tags.
        """
        combined_text = " ".join(text_samples[:3]) # Limit to top 3 to save context window
        
        if not self.model:
            return self._heuristic_fallback(combined_text)

        prompt = f"""<|system|>
You are a travel tagger. Classify the text into 1-2 categories from this list: {', '.join(self.valid_tags)}. Output as a comma-separated list.</s>
<|user|>
Text: {combined_text[:500]}...
Tags:</s>
<|assistant|>"""

        try:
            response = self.model(prompt, max_new_tokens=32, temperature=0.1)
            
            response_lower = response.lower()
            tags = set()
            
            for tag in self.valid_tags:
                if tag in response_lower:
                    tags.add(tag)
            
            return list(tags) if tags else ['general']
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            return self._heuristic_fallback(combined_text)

    def _heuristic_fallback(self, text):
        """Simulated logic if LLM is unavailable"""
        text = text.lower()
        tags = set()
        if 'mall' in text or 'shop' in text: tags.add('shopping')
        if 'waterfall' in text or 'nature' in text or 'park' in text: tags.add('nature')
        if 'temple' in text or 'church' in text: tags.add('spiritual')
        if 'food' in text or 'cafe' in text: tags.add('foodie')
        if 'trek' in text or 'hike' in text: tags.add('adventure')
        if 'museum' in text or 'fort' in text: tags.add('history')
        return list(tags) if tags else ['general']


class AttributeClassifier:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.eps = 0.5 
        self.cluster_model = DBSCAN(eps=self.eps, min_samples=2, metric='cosine')
        
        self.knn = None 
        self.llm = TinyLlamaWrapper()
        
        self.cluster_labels_map = {} 
        self.stored_ids = [] 
        
        self.is_fitted = False

    def fit_clusters(self, places_data: List[Dict]):
        """
        1. Embed all known places
        2. Run DBSCAN to find dense clusters (The "Themes")
        3. Use LLM to label each cluster center
        """
        if not places_data: return

        print(f"🧠 Training Hybrid Classifier on {len(places_data)} places...")
        
        texts = [f"{p['name']} {p.get('description', '')}" for p in places_data]
        embeddings = self.model.encode(texts)
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Run DBSCAN
        clusters = self.cluster_model.fit_predict(embeddings)
        
        unique_clusters = set(clusters)
        self.cluster_labels_map = {}
        
        for c_id in unique_clusters:
            if c_id == -1: continue # Skip noise
            
            indices = np.where(clusters == c_id)[0]
            samples = [texts[i] for i in indices[:5]] # Take top 5 examples
            
            generated_tags = self.llm.generate_tags(samples)
            self.cluster_labels_map[c_id] = generated_tags
            
        mask = clusters != -1
        if np.sum(mask) > 0:
            self.knn = NearestNeighbors(n_neighbors=1, metric='cosine')
            self.knn.fit(embeddings[mask])
            self.stored_ids = clusters[mask]
            self.is_fitted = True
            print(f"Discovered {len(self.cluster_labels_map)} themes: {self.cluster_labels_map}")
        else:
            print("⚠️ Not enough data to cluster yet.")

    def classify_user_type(self, query: str) -> str:
        """Determine user persona based on query"""
        q = query.lower()
        if 'trek' in q or 'adventure' in q: return 'Trekker'
        if 'food' in q or 'cafe' in q: return 'Foodie'
        if 'shop' in q or 'mall' in q: return 'Shopper'
        if 'temple' in q: return 'Pilgrim'
        return "General"

    def classify_with_rules(self, place: Dict) -> Dict:
        """
        Hybrid Inference:
        1. Embed new place
        2. Check if it falls into an existing Cluster (KNN)
        3. If yes -> Inherit Cluster Tags (Fast!)
        4. If no (Outlier) -> Use LLM directly (Slow path)
        """
        # Cold start check
        if not self.is_fitted:
            text = f"{place['name']} {place.get('description', '')}"
            tags = self.llm.generate_tags([text])
            return {"vibe": tags, "duration": ["half_day"], "best_time": ["any"]}

        text = f"{place['name']} {place.get('description', '')}"
        embedding = self.model.encode([text])
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        # Find nearest cluster member
        dist, idx = self.knn.kneighbors(embedding)
        
        # Threshold Check:
        # We use self.eps (0.5) to strictly match DBSCAN's definition of connectivity.
        # If distance < eps, it is "density reachable" from a known cluster point.
        if dist[0][0] < self.eps: 
            cluster_id = self.stored_ids[idx[0][0]]
            tags = self.cluster_labels_map.get(cluster_id, ['general'])
        else:
            # Outlier: Too far from any existing cluster to be part of it.
            # Use LLM to tag this specific unique place.
            tags = self.llm.generate_tags([text])
            
        return {
            "vibe": tags,
            "duration": ["half_day"],
            "best_time": ["day"]
        }
