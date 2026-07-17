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
import nltk
from nltk.corpus import stopwords
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

    @staticmethod
    def _extract_fallback_keywords(texts: list, top_n: int = 3) -> str:
        """
        Best-effort keyword extraction for cases where full BERTopic topic
        modeling doesn't run -- single-cluster fallback (doc_count < 15) or the
        BERTopic-failure fallback below. Previously these paths always labeled
        every cluster with the same static "General/Mixed Content" string
        regardless of actual content, which made every post's clusters
        indistinguishable from each other in the dashboard/debug dump. This
        does simple word-frequency counting over the given texts instead, so
        the label at least reflects what's actually being talked about. Falls
        back to "General/Mixed Content" only if no usable vocabulary remains
        (e.g. texts are entirely stopwords/emojis/single characters).
        """
        try:
            nltk.download('stopwords', quiet=True)
            multi_lang_stopwords = []
            for lang in ['arabic', 'english', 'french', 'spanish']:
                try:
                    multi_lang_stopwords.extend(stopwords.words(lang))
                except Exception:
                    pass

            vectorizer = CountVectorizer(
                min_df=1,
                stop_words=multi_lang_stopwords if multi_lang_stopwords else None,
                max_features=50
            )
            doc_term_matrix = vectorizer.fit_transform(texts)
            word_counts = doc_term_matrix.sum(axis=0).A1
            vocabulary = vectorizer.get_feature_names_out()

            top_indices = word_counts.argsort()[::-1][:top_n]
            top_words = [vocabulary[i] for i in top_indices if word_counts[i] > 0]

            return ", ".join(top_words) if top_words else "General/Mixed Content"
        except Exception as e:
            logger.warning(f"Fallback keyword extraction failed, using generic label: {e}")
            return "General/Mixed Content"

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

                similar_docs_list = []
                if len(community) > 1:
                    for idx in community[1:]:
                        similar_docs_list.append({
                            'comment_id': topic_comment_ids[idx],
                            'user_id': topic_user_ids[idx],
                            'text': topic_docs[idx]
                        })

                optimized_data.append({
                    'topic_id': int(topic_id),
                    'topic_keywords': keywords,
                    'representative_text': topic_docs[rep_idx],
                    'frequency': len(community),
                    'similar_docs': similar_docs_list, 
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
                        'similar_docs': [], 
                        'comment_id': topic_comment_ids[idx],
                        'user_id': topic_user_ids[idx]
                    })
                
        return optimized_data






    def _cluster_documents(self, documents: list) -> dict:
        """Runs the embedding + clustering pipeline for a single post's comments."""
        doc_count = len(documents)

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
            # Real BERTopic doesn't run for this small a doc count, so there's no
            # topic_keywords_dict to draw from. Extract a real word-frequency
            # summary of this post's own comments instead of leaving every post's
            # single cluster labeled with the same generic default.
            fallback_keywords = self._extract_fallback_keywords(texts)
            optimized_data = self.optimize_clusters(df, embeddings_tensor, topic_keywords_dict={0: fallback_keywords})
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
        nltk.download('stopwords', quiet=True)
        multi_lang_stopwords = []
        for lang in ['arabic', 'english', 'french', 'spanish']:
            try:
                multi_lang_stopwords.extend(stopwords.words(lang))
            except Exception:
                pass
        
        final_stopwords = multi_lang_stopwords if multi_lang_stopwords else None
        
        # token_pattern requires at least 2 word-characters by default, which already
        # drops most single-letter/number noise. min_df=1 keeps rare words instead of
        # dropping them, which is what actually causes "empty vocabulary" crashes when
        # a topic's comments are made up almost entirely of stopwords or very short
        # words (e.g. "for", "in", "on", "return", numbers, emojis-only text, etc).
        vectorizer_model = CountVectorizer(min_df=1, stop_words=final_stopwords)
        
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model
        )
        
        try:
            predicted_topics, _ = topic_model.fit_transform(texts, embeddings=embeddings_np)
        except ValueError as e:
            # Most commonly: "empty vocabulary; perhaps the documents only contain
            # stop words" -- happens when a topic ends up with comments that are
            # nothing but stopwords/short filler words after cleaning. Rather than
            # crashing the whole request, fall back to a single "General/Mixed
            # Content" cluster for this post.
            logger.warning(
                f"BERTopic/vectorizer failed ({e}); falling back to single-cluster "
                f"mode for this post instead of crashing."
            )
            df = pd.DataFrame({
                "text": texts,
                "comment_id": comment_ids,
                "user_id": user_ids,
                "Predicted_Topic": 0
            })
            fallback_keywords = self._extract_fallback_keywords(texts)
            optimized_data = self.optimize_clusters(df, embeddings_tensor, topic_keywords_dict={0: fallback_keywords})
            return {
                "num_clusters": 1,
                "noise_count": 0,
                "optimized_data": optimized_data
            }
        
        topic_keywords_dict = {}
        try:
            topic_info = topic_model.get_topic_info()
            if topic_info is not None:
                for _, row in topic_info.iterrows():
                    if row['Topic'] != -1:
                        try:
                            top_words = topic_model.get_topic(row['Topic'])
                            topic_keywords_dict[row['Topic']] = ", ".join(
                                [word for word, _ in top_words[:3]]
                            ) if top_words else "General/Mixed Content"
                        except Exception as e:
                            # Keyword extraction failing for a single topic shouldn't
                            # take down the whole response -- just label it generically.
                            logger.warning(f"Could not extract keywords for topic {row['Topic']}: {e}")
                            topic_keywords_dict[row['Topic']] = "General/Mixed Content"
        except Exception as e:
            logger.warning(f"Could not read topic info at all: {e}")
        
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

    def process(self, posts: list) -> dict:
        """
        Processes one or more posts. Each `post` must expose:
        page_name, post_id, post_user_id, post_content, comments (list of FBComment-like objects).
        Clustering runs independently per post (topics from different posts are never mixed
        together), then results are tagged with that post's metadata and combined.
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model is not loaded in memory")

        if not posts:
            raise ValueError("No posts were provided to process.")

        combined_optimized_data = []
        total_clusters = 0
        total_noise = 0
        failed_posts = []

        for post in posts:
            documents = post.comments
            logger.info(f"Processing post_id={post.post_id} ({len(documents)} comments)")

            try:
                result = self._cluster_documents(documents)
            except Exception as e:
                # Never let one bad post (weird encoding, degenerate text, etc.)
                # take down the whole multi-post request -- skip it and keep going.
                logger.error(f"Failed to process post_id={post.post_id}: {e}", exc_info=True)
                failed_posts.append(post.post_id)
                continue

            # Tag every cluster item with the post it came from, so downstream
            # formatting/routing can tell posts apart and avoid topic_id collisions.
            for item in result["optimized_data"]:
                item["page_name"] = post.page_name
                item["post_id"] = post.post_id

            combined_optimized_data.extend(result["optimized_data"])
            total_clusters += result["num_clusters"]
            total_noise += result["noise_count"]

        if failed_posts:
            logger.warning(f"The following posts failed and were skipped: {failed_posts}")

        return {
            "num_clusters": total_clusters,
            "noise_count": total_noise,
            "optimized_data": combined_optimized_data,
            "failed_posts": failed_posts
        }