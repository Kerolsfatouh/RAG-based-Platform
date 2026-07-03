import pandas as pd
import torch
import logging
import threading
import gc
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

class ClusteringPipeline:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading BAAI/bge-m3 on {self.device.upper()}")
        self.embedding_model = SentenceTransformer("BAAI/bge-m3", device=self.device)
        self.inference_lock = threading.Lock()

    def free_vram(self):
        if hasattr(self, 'embedding_model') and self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("BGE-M3 model forcefully removed from VRAM")

    def optimize_clusters(self, df: pd.DataFrame, embeddings: torch.Tensor, topic_keywords_dict: dict = None, similarity_threshold: float = 0.85) -> list:
        optimized_data = []
        if topic_keywords_dict is None:
            topic_keywords_dict = {}

        for topic_id in df['Predicted_Topic'].unique():
            if topic_id == -1:
                continue
                
            topic_indices = df[df['Predicted_Topic'] == topic_id].index.tolist()
            topic_docs = df.loc[topic_indices, 'text'].tolist()
            topic_comment_ids = df.loc[topic_indices, 'comment_id'].tolist()
            topic_user_ids = df.loc[topic_indices, 'user_id'].tolist()
            topic_embeddings = embeddings[topic_indices]
            
            communities = util.community_detection(
                topic_embeddings, 
                min_community_size=1, 
                threshold=similarity_threshold
            )
            
            covered_indices = set()
            keywords = topic_keywords_dict.get(topic_id, "General/Mixed Content")
            
            for community in communities:
                rep_idx = community[0]
                optimized_data.append({
                    'topic_id': int(topic_id),
                    'topic_keywords': keywords,
                    'representative_text': topic_docs[rep_idx],
                    'frequency': len(community),
                    'similar_docs_count': len(community) - 1,
                    'comment_id': topic_comment_ids[rep_idx],
                    'user_id': topic_user_ids[rep_idx]
                })
                covered_indices.update(community)
            
            missing_indices = set(range(len(topic_docs))) - covered_indices
            if missing_indices:
                for idx in missing_indices:
                    optimized_data.append({
                        'topic_id': int(topic_id),
                        'topic_keywords': keywords,
                        'representative_text': topic_docs[idx],
                        'frequency': 1,
                        'similar_docs_count': 0,
                        'comment_id': topic_comment_ids[idx],
                        'user_id': topic_user_ids[idx]
                    })
                
        return optimized_data

    def process(self, documents: list) -> dict:
        doc_count = len(documents)
        logger.info(f"Processing {doc_count} documents")
        
        if not self.embedding_model:
            raise RuntimeError("Embedding model is not loaded in memory")

        # Extract text and IDs from the FBComment objects
        texts = [doc.text for doc in documents]
        comment_ids = [doc.comment_id for doc in documents]
        user_ids = [doc.user_id for doc in documents]

        with self.inference_lock:
            embeddings_tensor = self.embedding_model.encode(texts, convert_to_tensor=True)
            embeddings_np = embeddings_tensor.cpu().numpy()

        if doc_count < 15:
            df = pd.DataFrame({
                "text": texts, 
                "comment_id": comment_ids, 
                "user_id": user_ids, 
                "Predicted_Topic": 0
            })
            optimized_data = self.optimize_clusters(df, embeddings_tensor)
            return {
                "num_clusters": 1,
                "noise_count": 0,
                "optimized_data": optimized_data
            }

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
        
        vectorizer_model = CountVectorizer(min_df=1)
        
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model
        )
        
        predicted_topics, _ = topic_model.fit_transform(texts, embeddings=embeddings_np)
        
        topic_keywords_dict = {}
        if topic_model.get_topic_info() is not None:
            topic_info = topic_model.get_topic_info()
            for _, row in topic_info.iterrows():
                if row['Topic'] != -1:
                    topic_keywords_dict[row['Topic']] = ", ".join([word for word, _ in topic_model.get_topic(row['Topic'])[:3]])
        
        df = pd.DataFrame({
            "text": texts, 
            "comment_id": comment_ids, 
            "user_id": user_ids, 
            "Predicted_Topic": predicted_topics
        })
        
        optimized_data = self.optimize_clusters(df, embeddings_tensor, topic_keywords_dict)
        optimized_data.sort(key=lambda x: (x['topic_id'], -x['frequency']))
        
        num_clusters = len(set(predicted_topics)) - (1 if -1 in predicted_topics else 0)
        noise_count = len(df[df['Predicted_Topic'] == -1])
        
        return {
            "num_clusters": num_clusters,
            "noise_count": noise_count,
            "optimized_data": optimized_data
        }