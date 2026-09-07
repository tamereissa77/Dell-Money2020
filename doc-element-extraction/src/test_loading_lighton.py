import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_loading():
    try:
        from model_handler_lighton import load_model, MODEL_CHECKPOINT
        import torch

        logger.info(f"Testing LightOnOCR model loading from: {MODEL_CHECKPOINT}")
        
        # Try to load just the processor first as a lightweight test
        from transformers import LightOnOcrProcessor
        logger.info("Loading processor...")
        # Note: In a real environment, this might download weights if not cached
        processor = LightOnOcrProcessor.from_pretrained(MODEL_CHECKPOINT)
        logger.info("Processor loaded successfully!")

        # Check if we should try loading the full model
        if len(sys.argv) > 1 and sys.argv[1] == "--full":
            logger.info("Loading full model (this may take a while and requires GPU)...")
            model, processor, device = load_model()
            logger.info(f"Model loaded successfully on {device}!")
        else:
            logger.info("Skipping full model load. Run with '--full' to test full model loading.")

        return True
    except Exception as e:
        logger.exception(f"Loading test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_loading()
    sys.exit(0 if success else 1)
