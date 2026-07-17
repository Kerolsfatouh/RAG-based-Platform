import logging
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# We use Chroma's default highly efficient embedding function for queries (all-MiniLM-L6-v2)
# This prevents us from having to load the massive BGE-M3 model every time someone asks a question!
embedding_fn = embedding_functions.DefaultEmbeddingFunction()


class VectorDB:
    def __init__(self):
        # Data will be saved permanently to data/chroma_db
        self.client = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="sports_clusters",
            embedding_function=embedding_fn
        )

    def is_empty(self) -> bool:
        """Lets callers check for data without paying for a query() call."""
        return self.collection.count() == 0

    def clear(self):
        """
        Wipes every cluster from the collection. Mirrors the old file-based pipeline's
        behavior of fully rebuilding daily_clusters.txt on every /update-knowledge call,
        so a fresh update doesn't leave stale clusters mixed in with new ones.
        Intended for deliberate manual resets, not the normal incremental update path
        (see delete_post() for that).
        """
        existing = self.collection.get()
        if existing and existing.get("ids"):
            self.collection.delete(ids=existing["ids"])

    def delete_post(self, post_id: str):
        """
        Removes every existing cluster belonging to a single post. Clustering isn't
        deterministic run-to-run (topic ids and cluster ordering can shift), so
        re-processing the same post and just upserting the new clusters can leave
        old, now-orphaned entries behind under different ids. Scoping the delete to
        just this post_id keeps every re-processed post internally consistent
        without wiping (or even touching) any other post's data.
        """
        try:
            self.collection.delete(where={"post_id": str(post_id)})
        except Exception as e:
            logger.warning(f"Could not delete existing clusters for post_id={post_id}: {e}")

    def add_clusters(self, clusters: list, posts_metadata: dict):
        ids = []
        documents = []
        metadatas = []

        for i, cluster in enumerate(clusters):
            post_id = cluster.get('post_id', 'unknown_post')
            topic_id = cluster.get('topic_id', '0')
            cluster_id = f"{post_id}_{topic_id}_{i}"

            meta = posts_metadata.get(post_id, {})
            page_name = meta.get('page_name', 'Unknown Page')
<<<<<<< HEAD
            post_content = meta.get('post_content', 'No content')
            
            # Format the exact text that the LLM will read
            content = f"Page: {page_name}\n"
            content += f"Post ID: {post_id}\n"
            content += f"Post Content: {post_content}\n"
            content += f"Topic/Intent: {cluster['topic_keywords']}\n"
            content += f"Main Comment [ID: {cluster.get('comment_id', 'N/A')} | User: {cluster.get('user_id', 'N/A')}]: {cluster['representative_text']}"
            
            if cluster.get('similar_docs'):
                content += "\nSimilar Comments in this thread:"
                for sub in cluster['similar_docs']:
                    content += f"\n- [ID: {sub.get('comment_id', 'N/A')} | User: {sub.get('user_id', 'N/A')}] {sub.get('text')}"
                    
=======
            intent = cluster['topic_keywords']

            # Kept separate from `similar_text` (instead of always merged) so that
            # search() can hand back both pieces and the caller decides -- based on the
            # router's requested depth -- whether to include the similar/sub-cluster
            # comments in the final answer context, same as the old regex-strip behavior.
            main_text = (
                f"[Main] [Comment ID: {cluster.get('comment_id', 'N/A')} | "
                f"User ID: {cluster.get('user_id', 'N/A')}] {cluster['representative_text']}"
            )

            similar_text = ""
            if cluster.get('similar_docs'):
                sub_lines = [
                    f"- [{sub.get('comment_id')} | {sub.get('user_id')}] {sub.get('text')}"
                    for sub in cluster['similar_docs']
                ]
                similar_text = "\n".join(sub_lines)

            # The embedded/searched document still includes everything, so recall isn't
            # hurt by hiding the similar comments -- only the *answer context* respects depth.
            content = f"Page: {page_name}\nTopic/Intent: {intent}\nMain Comment: {cluster['representative_text']}"
            if similar_text:
                content += "\nSimilar Comments in this thread:\n" + similar_text

>>>>>>> 5cffe7f (refactor: improve RAG pipeline stability)
            documents.append(content)
            metadatas.append({
                "post_id": str(post_id),
                "page_name": str(page_name),
                "intent": str(intent),
                "main_text": main_text,
                "similar_text": similar_text,
            })
            ids.append(str(cluster_id))

        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, n_results: int = 4):
        """
        Turns the query into a vector and fetches the best matching clusters.
        Returns a list of candidate dicts (not raw strings) so the caller -- QwenEngine --
        can reconstruct the final context text itself (with or without similar/sub-cluster
        comments, depending on the requested depth) and still knows the real cluster id
        for each result instead of just a positional index.
        """
        if self.collection.count() == 0:
            return []

        # Ensure we don't ask for more results than we have in the DB
        n_results = min(n_results, self.collection.count())

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        if not results.get('ids') or not results['ids'][0]:
            return []

        candidates = []
        for idx, chroma_id in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][idx]
            candidates.append({
                "id": chroma_id,
                "post_id": meta.get("post_id"),
                "page_name": meta.get("page_name"),
                "intent": meta.get("intent"),
                "main_text": meta.get("main_text", ""),
                "similar_text": meta.get("similar_text", ""),
            })
        return candidates
