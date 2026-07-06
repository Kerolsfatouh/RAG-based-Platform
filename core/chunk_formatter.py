def format_clusters_to_chunks(optimized_data: list, posts_metadata: dict) -> str:
    """
    optimized_data: list of cluster items, each tagged with 'post_id' and 'page_name'
                    (this happens automatically in ClusteringPipeline.process()).
    posts_metadata: dict keyed by post_id -> {"page_name": ..., "post_content": ...}
                    Build this from the same list of posts you passed into the pipeline, e.g.:
                        posts_metadata = {
                            p.post_id: {"page_name": p.page_name, "post_content": p.post_content}
                            for p in comment_input.posts
                        }
    """
    grouped_topics = {}
    group_order = []

    for item in optimized_data:
        # Group by (post_id, topic_id) instead of topic_id alone, since topic ids
        # are only unique *within* a single post's clustering run, not across posts.
        post_id = item.get('post_id', 'unknown')
        group_key = (post_id, item['topic_id'])

        if group_key not in grouped_topics:
            grouped_topics[group_key] = {
                "keywords": item['topic_keywords'],
                "contents": [],
                "post_id": post_id,
            }
            group_order.append(group_key)

        main_line = f"- [Main] [Comment ID: {item.get('comment_id', 'N/A')} | User ID: {item.get('user_id', 'N/A')}] {item['representative_text']}"

        if item.get('similar_docs'):
            sub_lines = "\n  |_ [Sub-cluster Hidden]"
            for sub in item['similar_docs']:
                sub_lines += f"\n      - [{sub.get('comment_id')} | {sub.get('user_id')}] {sub.get('text')}"
            main_line += sub_lines

        grouped_topics[group_key]["contents"].append(main_line)

    final_chunks = []
    # Re-number clusters sequentially (0, 1, 2...) across ALL posts combined, so
    # CLUSTER_ID stays a plain unique integer for qwen_engine's router/regex logic,
    # even though the underlying clusters come from several different posts.
    for global_cluster_id, group_key in enumerate(group_order):
        data = grouped_topics[group_key]
        post_id = data["post_id"]
        post_meta = posts_metadata.get(post_id, {})

        chunk = f"<CLUSTER_START>\n"
        chunk += f"[CLUSTER_ID: {global_cluster_id}]\n"
        chunk += f"[FACEBOOK_PAGE: {post_meta.get('page_name', 'N/A')}]\n"
        chunk += f"[POST_ID: {post_id}]\n"
        chunk += f"[POST_CONTENT: {post_meta.get('post_content', '')}]\n"
        chunk += f"[TOP_INTENT: {data['keywords']}]\n"
        chunk += f"[CONTENT]:\n" + "\n".join(data['contents']) + "\n"
        chunk += f"<CLUSTER_END>"
        final_chunks.append(chunk)

    return "\n\n".join(final_chunks)