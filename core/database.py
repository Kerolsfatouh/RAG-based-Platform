import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

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
        
    def add_clusters(self, clusters: list, posts_metadata: dict):
        ids = []
        documents = []
        metadatas = []
        
        for i, cluster in enumerate(clusters):
            post_id = cluster.get('post_id', 'unknown_post')
            cluster_id = f"{post_id}_{cluster.get('topic_id', '0')}_{i}"
            
            meta = posts_metadata.get(post_id, {})
            page_name = meta.get('page_name', 'Unknown Page')
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
                    
            documents.append(content)
            metadatas.append({
                "post_id": str(post_id),
                "page_name": str(page_name),
                "intent": str(cluster['topic_keywords'])
            })
            ids.append(str(cluster_id))
            
        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            
    def search(self, query: str, n_results: int = 4):
        """Turn the query into a vector and instantly fetch the best matching clusters"""
        if self.collection.count() == 0:
            return []
            
        # Ensure we don't ask for more results than we have in the DB
        n_results = min(n_results, self.collection.count())
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        return results['documents'][0]
