import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

class LocalQwenLLM:
    """
    A Langchain-compatible wrapper for our locally fine-tuned Qwen2.5-VL model.
    This mimics the ChatGoogleGenerativeAI class so it works seamlessly with extract_documents.py.
    """
    def __init__(self):
        # Point to your newly trained best checkpoint
        self.model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        # Adjust this path if you move this file into loan_processing!
        self.best_dir = os.path.join(os.path.dirname(__file__), "output", "checkpoints", "best")
        
        if not os.path.exists(self.best_dir):
            raise FileNotFoundError(f"Best adapter not found at {self.best_dir}. Did training finish?")

        print(f"[LocalQwenLLM] Loading fine-tuned Native Vision model into VRAM...")
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, self.best_dir)
        self.model.eval()
        print("[LocalQwenLLM] Model loaded successfully!")

    def invoke(self, messages):
        """
        Takes Langchain HumanMessage objects and converts them to Qwen2.5-VL format.
        """
        qwen_messages = []
        
        if isinstance(messages, str):
            qwen_messages.append({"role": "user", "content": [{"type": "text", "text": messages}]})
        elif isinstance(messages, list):
            for msg in messages:
                # Extract role and content whether it's a dict or Langchain message
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    raw_content = msg.get("content", [])
                else:
                    role = "user"
                    raw_content = getattr(msg, "content", [])

                # Normalize content to Qwen format
                qwen_content = []
                if isinstance(raw_content, str):
                    qwen_content.append({"type": "text", "text": raw_content})
                elif isinstance(raw_content, list):
                    for part in raw_content:
                        if not isinstance(part, dict):
                            qwen_content.append({"type": "text", "text": str(part)})
                            continue
                            
                        if part.get("type") == "text":
                            qwen_content.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            # Map Langchain/OpenAI "image_url" -> Qwen "image"
                            img_url = part.get("image_url", {})
                            if isinstance(img_url, dict):
                                img_url = img_url.get("url", "")
                            qwen_content.append({"type": "image", "image": img_url, "max_pixels": 800000})
                        elif part.get("type") == "image":
                            qwen_content.append({"type": "image", "image": part.get("image", ""), "max_pixels": 800000})
                
                qwen_messages.append({"role": role, "content": qwen_content})

        text = self.processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([qwen_messages])
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=200,
                repetition_penalty=1.0,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            
        response_text = self.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Create a mock object that mimics Langchain's response so _get_text() works
        class MockResponse:
            def __init__(self, content):
                self.content = content
                
        return MockResponse(response_text)

# We use a global singleton so that the 14GB model isn't reloaded for every single document!
_local_llm_instance = None

def get_llm():
    """Return the locally fine-tuned Qwen2.5-VL instance instead of Gemini."""
    global _local_llm_instance
    if _local_llm_instance is None:
        _local_llm_instance = LocalQwenLLM()
    return _local_llm_instance
