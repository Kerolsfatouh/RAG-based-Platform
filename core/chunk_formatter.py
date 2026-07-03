def format_clusters_to_chunks(optimized_data: list, metadata: dict) -> str:
    grouped_topics = {}
    for item in optimized_data:
        t_id = item['topic_id']
        if t_id not in grouped_topics:
            grouped_topics[t_id] = {
                "keywords": item['topic_keywords'],
                "contents": []
            }
        grouped_topics[t_id]["contents"].append(
            f"- [Comment ID: {item.get('comment_id', 'N/A')} | User ID: {item.get('user_id', 'N/A')}] {item['representative_text']} (Frequency: {item['frequency']})"
        )

    final_chunks = []
    for t_id, data in grouped_topics.items():
        chunk = f"<CLUSTER_START>\n"
        chunk += f"[CLUSTER_ID: {t_id}]\n"
        chunk += f"[FACEBOOK_PAGE: {metadata['page_name']}]\n"
        chunk += f"[POST_ID: {metadata['post_id']}]\n"
        chunk += f"[POST_CONTENT: {metadata['post_content']}]\n"
        chunk += f"[TOP_INTENT: {data['keywords']}]\n"
        chunk += f"[CONTENT]:\n" + "\n".join(data['contents']) + "\n"
        chunk += f"<CLUSTER_END>"
        final_chunks.append(chunk)
        
    return "\n\n".join(final_chunks)