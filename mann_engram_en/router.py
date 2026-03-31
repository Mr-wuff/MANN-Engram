import torch
import torch.nn.functional as F
import numpy as np
import io
import base64
from PIL import Image
from typing import List, Dict, Tuple, Union, Any, Optional
from transformers import AutoProcessor, AutoModel
from openai import OpenAI

# Import internal modules
from .models import SkewGaussian
from .document_parser import DocumentParser
from .intent_extractor import IntentExtractor

class MANNEngramRouter:
    """
    MANN-Engram: Edge-Cloud Multimodal Semantic Router.
    Acts as the main orchestrator for Intent Extraction, Document Parsing, and Tensor Routing.
    """
    def __init__(self, 
                 siglip_model_path: str = "google/siglip-so400m-patch14-384", 
                 ckpt_path: str = "skew_model_v4full.pt", 
                 device: str = None,
                 enable_local_intent: bool = False,
                 local_intent_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
                 
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MANN-Engram] Initializing on {self.device.upper()}...")
        
        # 1. Load Intent Engine
        self.intent_extractor = IntentExtractor(
            device=self.device, 
            enable_local=enable_local_intent, 
            local_model_id=local_intent_model
        )
        
        # 2. Load Vision-Language Encoders
        self.processor = AutoProcessor.from_pretrained(siglip_model_path)
        self.siglip_model = AutoModel.from_pretrained(siglip_model_path).to(self.device)
        self.siglip_model.eval()
        
        # 3. Load MANN Routing Core
        self.compressor = SkewGaussian().to(self.device)
        self.compressor.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        self.compressor.eval()
        print("[MANN-Engram] Tensor Routing Engines warmed up successfully.")

    def _encode_texts(self, text_list: List[str]) -> torch.Tensor:
        if not text_list: return torch.empty((0, 1152)).to(self.device)
        inputs = self.processor(text=text_list, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(self.device)
        clean_inputs = {"input_ids": inputs["input_ids"]}
        if "attention_mask" in inputs: clean_inputs["attention_mask"] = inputs["attention_mask"]
        
        with torch.no_grad():
            try: 
                features = self.siglip_model.get_text_features(**clean_inputs)
            except AttributeError: 
                outputs = self.siglip_model.text_model(**clean_inputs)
                pooled = outputs[1] if isinstance(outputs, tuple) else outputs.pooler_output
                features = self.siglip_model.text_projection(pooled)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features

    def _encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        if not pil_images: return torch.empty((0, 1152)).to(self.device)
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.siglip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features

    def _top_p_sampling(self, scores: np.ndarray, items: List[Any], top_p: float, temperature: float = 0.5) -> Tuple[List[Any], List[float]]:
        if len(scores) == 0: return [], []
        t_scores = torch.tensor(scores)
        probs = F.softmax((t_scores - t_scores.max()) / temperature, dim=0).numpy()
        sorted_idx = np.argsort(probs)[::-1]
        
        selected_items, selected_probs, cum_p = [], [], 0.0
        for idx in sorted_idx:
            prob = float(probs[idx])
            cum_p += prob
            selected_items.append(items[idx])
            selected_probs.append(prob)
            if cum_p >= top_p: break
        return selected_items, selected_probs

    def compress(self, query: str, context_pool: List[str] = [], image_pool: List[Union[str, Image.Image]] = [], top_p: float = 0.85) -> Dict[str, Any]:
        """Low-level method: Evaluates strictly defined text blocks and images against a pure query."""
        pil_images = [Image.open(img).convert("RGB") if isinstance(img, str) else img.convert("RGB") for img in image_pool]
        q_pure_emb = self._encode_texts([query])
        
        valid_images, image_probs = [], []
        if pil_images:
            img_embs = self._encode_images(pil_images)
            with torch.no_grad():
                img_scores = self.compressor.compute_score(q_pure_emb, img_embs).cpu().numpy().flatten()
            valid_images, image_probs = self._top_p_sampling(img_scores, pil_images, top_p)
            valid_idx = [pil_images.index(img) for img in valid_images]
            q_probe = F.normalize(q_pure_emb + img_embs[valid_idx].mean(dim=0, keepdim=True), p=2, dim=-1)
        else:
            q_probe = q_pure_emb

        valid_texts, text_probs = [], []
        if context_pool:
            c_text_embs = self._encode_texts(context_pool).unsqueeze(0)
            with torch.no_grad():
                text_scores = self.compressor.compute_score(q_probe, c_text_embs).cpu().numpy().flatten()
            valid_texts, text_probs = self._top_p_sampling(text_scores, context_pool, top_p)

        b64_images = []
        for img in valid_images:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            b64_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))

        return {
            "core_query": query,
            "purified_context": "\n\n".join(valid_texts),
            "purified_images_b64": b64_images,
            "purified_images_pil": valid_images,
            "stats": {
                "original_images": len(pil_images), "retained_images": len(valid_images),
                "original_text_chunks": len(context_pool), "retained_text_chunks": len(valid_texts)
            }
        }

    def process_session(self, 
                        raw_chat_input: str, 
                        file_paths: List[str] = [], 
                        image_paths: List[str] = [],
                        openai_client: Optional[OpenAI] = None,
                        intent_model: str = "gpt-4o-mini",
                        top_p: float = 0.85) -> Dict[str, Any]:
        """
        High-level End-to-End Pipeline.
        """
        print("[MANN-Engram] Step 1: Parsing user intent...")
        parsed_intent = self.intent_extractor.parse_user_input(raw_chat_input, openai_client, intent_model)
        core_query = parsed_intent['query']
        
        print("[MANN-Engram] Step 2: Parsing uploaded files...")
        context_pool = []
        if parsed_intent['context']:
            context_pool.append(f"[User Chat Context]: {parsed_intent['context']}")
            
        for f_path in file_paths:
            chunks = DocumentParser.parse_file(f_path)
            context_pool.extend(chunks)
            
        print(f"[MANN-Engram] Generated {len(context_pool)} context chunks. Step 3: Semantic Routing...")
        return self.compress(query=core_query, context_pool=context_pool, image_pool=image_paths, top_p=top_p)