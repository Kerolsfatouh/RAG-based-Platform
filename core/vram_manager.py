import gc
import torch
import logging

logger = logging.getLogger(__name__)

def clear_vram():
    """Clears system RAM and GPU VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.info("System RAM and GPU VRAM have been completely cleared!")