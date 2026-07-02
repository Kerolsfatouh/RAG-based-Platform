import pandas as pd
import torch
import logging
import threading
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

class ClusteringPipeline:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading BAAI/bge-m3 on {self.device.upper()}...")
        self.embedding_model = SentenceTransformer("BAAI/bge-m3", device=self.device)
        
        # Lock to ensure only one inference at a time, preventing memory/VRAM issues
        self.inference_lock = threading.Lock()

    def optimize_clusters(self, df: pd.DataFrame, embeddings: torch.Tensor, similarity_threshold: float = 0.85) -> list:
        optimized_data = []
        for topic_id in df['Predicted_Topic'].unique():
            if topic_id == -1:
                continue
                
            topic_indices = df[df['Predicted_Topic'] == topic_id].index.tolist()
            topic_docs = df.loc[topic_indices, 'text'].tolist()
            topic_embeddings = embeddings[topic_indices]
            
            communities = util.community_detection(
                topic_embeddings, 
                min_community_size=1, 
                threshold=similarity_threshold
            )
            
            # track which comments have been covered by communities to handle isolated docs
            covered_indices = set()
            
            for community in communities:
                rep_idx = community[0]
                optimized_data.append({
                    'topic_id': int(topic_id),
                    'representative_text': topic_docs[rep_idx],
                    'frequency': len(community),
                    'similar_docs_count': len(community) - 1
                })
                covered_indices.update(community)
            
            # Fallback Mechanism
            missing_indices = set(range(len(topic_docs))) - covered_indices
            if missing_indices:
                logger.warning(f"Topic {topic_id}: Recovering {len(missing_indices)} isolated docs dropped by community_detection.")
                for idx in missing_indices:
                    optimized_data.append({
                        'topic_id': int(topic_id),
                        'representative_text': topic_docs[idx],
                        'frequency': 1,
                        'similar_docs_count': 0
                    })
                
        return optimized_data

    def process(self, documents: list) -> dict:
        doc_count = len(documents)
        logger.info(f"Processing {doc_count} documents...")
        
        # handle the inference lock to ensure only one request is performing inference at a time
        with self.inference_lock:
            logger.info("Computing embeddings sequentially to protect Memory/VRAM...")
            embeddings_tensor = self.embedding_model.encode(documents, convert_to_tensor=True)
            embeddings_np = embeddings_tensor.cpu().numpy()

        # Bypass BERTopic if doc count < 5
        if doc_count < 5:
            logger.info("Doc count < 5. Bypassing BERTopic and routing to direct deduplication.")
            df = pd.DataFrame({"text": documents, "Predicted_Topic": 0})
            optimized_data = self.optimize_clusters(df, embeddings_tensor)
            return {
                "num_clusters": 1,
                "noise_count": 0,
                "optimized_data": optimized_data
            }

        # full isolation of objects to ensure thread-safety and no state leakage
        n_neighbors = max(2, min(15, doc_count // 3))
        min_cluster_size = max(2, min(15, doc_count // 3))
        
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=min(5, doc_count-2), metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size, 
            min_samples=max(1, min_cluster_size // 3), 
            cluster_selection_epsilon=0.4,
            metric='euclidean', 
            cluster_selection_method='eom', 
            prediction_data=True
        )
        vectorizer_model = CountVectorizer(min_df=1 if doc_count < 10 else 2, max_df=0.9)
        
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model
        )
        
        logger.info("Fitting BERTopic model...")
        predicted_topics, _ = topic_model.fit_transform(documents, embeddings=embeddings_np)
        
        df = pd.DataFrame({"text": documents, "Predicted_Topic": predicted_topics})
        
        logger.info("Optimizing clusters for RAG...")
        optimized_data = self.optimize_clusters(df, embeddings_tensor)
        optimized_data.sort(key=lambda x: (x['topic_id'], -x['frequency']))
        
        num_clusters = len(set(predicted_topics)) - (1 if -1 in predicted_topics else 0)
        noise_count = len(df[df['Predicted_Topic'] == -1])
        
        return {
            "num_clusters": num_clusters,
            "noise_count": noise_count,
            "optimized_data": optimized_data
        }