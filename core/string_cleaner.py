import re
import json

def clean_text_for_json(text: str) -> str:
    if not text:
        return ""
    
    # Remove hidden or corrupted control characters except standard newlines/tabs
    text = re.sub(re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"), "", text)
    
    # Normalize multiple consecutive newlines or tabs into single spaces to avoid formatting breaking
    text = re.sub(r"[\r\n\t]+", " ", text)
    
    # Strip leading/trailing whitespaces
    text = text.strip()
    
    # Use json.dumps to automatically escape quotes, backslashes, and special unicode chars safely
    safe_json_str = json.dumps(text, ensure_ascii=False)
    
    # Return the clean text without the surrounding quotes added by json.dumps
    return safe_json_str[1:-1]