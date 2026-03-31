import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import traceback
import base64
import os
import json
import glob
import io
from PIL import Image
from transformers import AutoProcessor, AutoModel
from openai import OpenAI

# ==========================================
# 0. Page Config & Scenarios (Dynamic Loader)
# ==========================================
st.set_page_config(page_title="MANN-Engram | Multimodal Semantic Router", layout="wide", page_icon="🧠")

IMAGE_DIR = "VQAMed2019_Test_Images"

def load_local_cases():
    """Dynamically load cases from JSON files in the current directory, supporting both ER Disaster and MDT formats."""
    cases = {}
    search_path = os.path.join("cases", "*.json")
    json_files = glob.glob(search_path)
    for f in json_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Format A: MDT Case Format (e.g., MDT_Case_01_EN.json)
                if 'mdt_case' in data:
                    mdt = data['mdt_case']
                    title = f"MDT Case {mdt.get('case_id', '')}: {mdt.get('title', f)}"
                    
                    # Parse Profile
                    profile = mdt.get('patient_profile', {})
                    context = "=== PATIENT PROFILE ===\n"
                    context += f"Age/Gender: {profile.get('age', 'N/A')} / {profile.get('gender', 'N/A')}\n"
                    context += f"Chief Complaint: {profile.get('chief_complaint', 'N/A')}\n"
                    context += f"HPI: {profile.get('history_of_present_illness', 'N/A')}\n"
                    context += f"PMH: {', '.join(profile.get('past_medical_history', []))}\n\n"
                    
                    # Parse Transcript Log
                    context += "=== EHR & MDT TRANSCRIPT LOG ===\n"
                    for log in mdt.get('ehr_and_transcript_log', []):
                        context += f"[{log.get('timestamp', '')}] {log.get('role', '')}: {log.get('content', '')}\n\n"
                        
                    # Parse Images
                    img_repo = mdt.get('image_repository', [])
                    image_filenames = [img.get('file_name', '') for img in img_repo if img.get('file_name')]
                    
                    img_hints = f"Expected Images in '{IMAGE_DIR}/':\n"
                    img_hints += "\n".join([f"- {img.get('file_name', '')}: {img.get('role', '')}" for img in img_repo])
                    
                    query = "You are a top MDT expert. Synthesize the provided multidisciplinary data and multimodal images. Identify the primary diagnosis, explain the sequence of events (pathophysiology), and recommend the immediate next steps."
                    
                    cases[title] = {
                        "query": query,
                        "context": context,
                        "img_hints": img_hints,
                        "image_filenames": image_filenames
                    }
                
                # Format B: ER Disaster Format (e.g., er_disaster_cases.json)
                else:
                    for key, val in data.items():
                        if isinstance(val, dict) and "query" in val and "context" in val:
                            img_hints = val.get("img_hints", "")
                            
                            # Use regex to automatically extract filenames from the img_hints text
                            image_filenames = re.findall(r'[a-zA-Z0-9_-]+\.(?:jpg|jpeg|png)', img_hints)
                            
                            cases[key] = {
                                "query": val["query"],
                                "context": val["context"],
                                "img_hints": img_hints,
                                "image_filenames": image_filenames
                            }
                            
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")
    return cases

# Combine hardcoded fallbacks and dynamically loaded cases
SCENARIOS = {
    "Hell-Mode Fallback: ER Disaster (Hidden Fatal Lesion)": {
        "query": "Ignore the patient's irrelevant complaints. Based on the multimodal data, what is the most critical and fatal acute condition? Provide a preliminary diagnosis and immediate life-saving actions.",
        "context": "[Patient Self-Complaint]: Doctor, finally! I have been waiting for 3 hours... (Fallback Data)",
        "img_hints": "No local images specified for fallback.",
        "image_filenames": []
    }
}

local_cases = load_local_cases()
if local_cases:
    SCENARIOS.update(local_cases)

# Initialize Session State
if 'current_scenario' not in st.session_state or st.session_state.current_scenario not in SCENARIOS:
    st.session_state.current_scenario = list(SCENARIOS.keys())[-1] # Default to the last loaded real case
    st.session_state.query_text = SCENARIOS[st.session_state.current_scenario]["query"]
    st.session_state.context_text = SCENARIOS[st.session_state.current_scenario]["context"]

with st.sidebar:
    st.header("☁️ Cloud VLM API")
    API_BASE_URL = st.text_input("Base URL", value="https://api.siliconflow.cn/v1")
    API_KEY = st.text_input("API Key", type="password", placeholder="sk-...")
    MODEL_NAME = st.text_input("Model Name", value="Qwen/Qwen2-VL-7B-Instruct")
    
    st.divider()
    st.header("📚 Load MDT & ER Scenarios")
    selected_scenario = st.selectbox("Select Scenario", list(SCENARIOS.keys()), index=list(SCENARIOS.keys()).index(st.session_state.current_scenario))
    
    if selected_scenario != st.session_state.current_scenario:
        st.session_state.current_scenario = selected_scenario
        st.session_state.query_text = SCENARIOS[selected_scenario]["query"]
        st.session_state.context_text = SCENARIOS[selected_scenario]["context"]
        st.rerun()
        
    st.info(f"📸 **Image Hints**: \n\n{SCENARIOS[selected_scenario]['img_hints']}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_SIGLIP_PATH = "./siglip-so400m"

# ==========================================
# 1. MANN-Engram Core (Local Middleware)
# ==========================================
class BackboneNet(nn.Module):
    def __init__(self, d=1152, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden))
    def forward(self, x): return self.net(x)

class SkewGaussian(nn.Module):
    def __init__(self, d=1152, rank=16, hidden=256):
        super().__init__()
        self.backbone = BackboneNet(d, hidden)
        self.head_d, self.head_L = nn.Linear(hidden, d), nn.Linear(hidden, d * rank)
        self.head_delta, self.head_alpha = nn.Linear(hidden, d), nn.Linear(hidden, d)

    def forward(self, q):
        h = self.backbone(q)
        return self.head_d(h), self.head_L(h), self.head_delta(h), self.head_alpha(h)

    def compute_score(self, q, c, ctx_mask=None):
        log_d, L_flat, delta, alpha = self.forward(q)
        log_d = torch.clamp(log_d, min=-15.0, max=15.0) 
        B, D = q.shape
        R = L_flat.shape[-1] // D
        L = L_flat.view(B, D, R)
        center = q + delta
        diff = c - center.unsqueeze(1)
        d_val = torch.exp(log_d)
        d_inv = 1.0 / d_val
        mahal = torch.sum(diff**2 * d_inv.unsqueeze(1), dim=-1)
        DinvL = d_inv.unsqueeze(-1) * L
        M = torch.eye(R, device=q.device).unsqueeze(0) + torch.bmm(L.transpose(1, 2), DinvL)
        Mi = torch.linalg.inv(M + 1e-4 * torch.eye(R, device=q.device).unsqueeze(0)) 
        v = diff * d_inv.unsqueeze(1)
        w = torch.bmm(L.transpose(1, 2), v.transpose(1, 2))
        mahal = mahal - torch.sum(w * torch.bmm(Mi, w), dim=1)
        ld = torch.sum(log_d, dim=-1, keepdim=True) + torch.linalg.slogdet(M)[1].unsqueeze(-1)
        lp = -0.5 * (mahal + ld) + F.logsigmoid(torch.sum(alpha.unsqueeze(1) * diff, dim=-1))
        return lp

@st.cache_resource(show_spinner="Warming up MANN-Engram Tensor Engine...")
def init_engines():
    processor = AutoProcessor.from_pretrained(LOCAL_SIGLIP_PATH, local_files_only=True)
    siglip = AutoModel.from_pretrained(LOCAL_SIGLIP_PATH, local_files_only=True).to(DEVICE)
    siglip.eval()
    compressor = SkewGaussian().to(DEVICE)
    compressor.load_state_dict(torch.load("skew_model_v4full.pt", map_location=DEVICE, weights_only=True))
    compressor.eval()
    return processor, siglip, compressor

processor, siglip_model, compressor_model = init_engines()

def encode_text_siglip(text_list):
    if not text_list: return torch.empty((0, 1152)).to(DEVICE)
    inputs = processor(text=text_list, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
    clean_inputs = {"input_ids": inputs["input_ids"]}
    if "attention_mask" in inputs: clean_inputs["attention_mask"] = inputs["attention_mask"]
    with torch.no_grad():
        try: text_features = siglip_model.get_text_features(**clean_inputs)
        except: 
            text_outputs = siglip_model.text_model(**clean_inputs)
            pooled_output = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs.pooler_output
            text_features = siglip_model.text_projection(pooled_output)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features

def encode_image_siglip(pil_images):
    """Expects a list of PIL Image objects."""
    if not pil_images: return torch.empty((0, 1152)).to(DEVICE)
    inputs = processor(images=pil_images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_features = siglip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features

def image_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_cloud_vlm(prompt_text, base64_images=[]):
    if not API_KEY: return "⚠️ Error: API Key missing.", 0, 0
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    content_list = [{"type": "text", "text": prompt_text}]
    for b64 in base64_images:
        if b64: content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    messages = [{"role": "user", "content": content_list}]
    try:
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=4096, temperature=0.2)
        content = response.choices[0].message.content
        pt = response.usage.prompt_tokens if response.usage else 0
        ct = response.usage.completion_tokens if response.usage else 0
        return content, pt, ct
    except Exception as e:
        return f"🚨 API Error: {str(e)}", 0, 0

# Top-P Nucleus Sampling
def apply_top_p_filtering(scores, items, top_p_threshold, temperature=0.5):
    if len(scores) == 0: return [], ""
    t_scores = torch.tensor(scores)
    probs = F.softmax((t_scores - t_scores.max()) / temperature, dim=0).numpy()
    sorted_idx = np.argsort(probs)[::-1]
    
    sel_items, log_output, cum_p = [], "", 0.0
    for i, idx in enumerate(sorted_idx):
        prob = probs[idx]
        cum_p += prob
        sel_items.append(items[idx])
        log_output += f"**[Adaptive K={i+1}]** (Prob: {prob*100:.1f}%, Cumulative: {cum_p*100:.1f}%): {str(items[idx])[:80]}...\n\n"
        if cum_p >= top_p_threshold:
            log_output += f"*(ℹ️ Reached Top-P {top_p_threshold*100:.0f}%, truncated remaining noise.)*\n"
            break
    return sel_items, log_output

# ==========================================
# 3. UI Layout
# ==========================================
st.title("🧠 MANN-Engram: Edge-Cloud Multimodal Router")
st.markdown("*Dual-Routing & Top-P Nucleus Sampling to Shield Cloud VLMs from Information Tsunamis*")
st.divider()

col1, col2 = st.columns([5, 6])

with col1:
    st.markdown("### 📥 Diagnostic Input")
    query = st.text_area("1. Query", value=st.session_state.query_text, height=80)
    
    st.markdown(f"**2. 📎 Loaded Images from Current Scenario**")
    scenario_data = SCENARIOS[st.session_state.current_scenario]
    expected_imgs = scenario_data.get("image_filenames", [])
    
    local_images = []
    local_image_names = []
    missing_images = []
    
    # Load images directly from disk
    for fname in expected_imgs:
        img_path = os.path.join(IMAGE_DIR, fname)
        if os.path.exists(img_path):
            local_images.append(Image.open(img_path).convert("RGB"))
            local_image_names.append(fname)
        else:
            missing_images.append(fname)
            
    if local_images:
        st.success(f"Successfully loaded {len(local_images)} images from '{IMAGE_DIR}'.")
        if missing_images:
            st.warning(f"Missing images in directory: {', '.join(missing_images)}")
            
        cols = st.columns(4)
        for idx, img in enumerate(local_images):
            with cols[idx % 4]: 
                st.image(img, caption=local_image_names[idx], use_container_width=True)
    else:
        st.info(f"No automatically loaded images found in '{IMAGE_DIR}'. You can upload them manually below.")
        
    uploaded_files = st.file_uploader("Or Upload Additional Images (Optional)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    # Combine local and uploaded images
    all_images = list(local_images)
    all_image_names = list(local_image_names)
    if uploaded_files:
        for f in uploaded_files:
            all_images.append(Image.open(f).convert("RGB"))
            all_image_names.append(f.name)
            
    context_text = st.text_area("3. Redundant Context Pool (Noisy Data)", value=st.session_state.context_text, height=350)
    
    st.markdown("**4. Latent Space Hyperparameter**")
    top_p_val = st.slider("Top-P Nucleus Threshold", 0.3, 0.99, 0.85, 0.05, help="Automatically truncate when cumulative probability of selected evidence reaches this threshold.")
    
    submit_btn = st.button("🚀 Trigger MANN-Engram Extractor", type="primary", use_container_width=True)

with col2:
    st.markdown("### 🔭 Latent Space Routing Radar")
    log_area = st.empty()
    result_area = st.container()

# ==========================================
# 4. Execution Logic
# ==========================================
if submit_btn:
    if not API_KEY: st.error("API Key required!"); st.stop()
    try:
        text_contexts = [c.strip() for c in re.split(r'\n\s*\n', context_text) if c.strip()]
        if not text_contexts: st.stop()

        q_pure_text_emb = encode_text_siglip([query])
        valid_images = []
        log_full = "✨ **[Phase 1: Visual Anti-Spoofing] (Text -> Image)**\n\n"
        
        if all_images:
            img_embs = encode_image_siglip(all_images)
            with torch.no_grad():
                img_scores = compressor_model.compute_score(q_pure_text_emb, img_embs).cpu().numpy().flatten()
            
            sel_names, img_log = apply_top_p_filtering(img_scores, all_image_names, top_p_val)
            log_full += img_log + "---\n\n"
            
            valid_images = [img for img, name in zip(all_images, all_image_names) if name in sel_names]
            valid_idx = [i for i, name in enumerate(all_image_names) if name in sel_names]
            valid_img_embs = img_embs[valid_idx]
            
            q_probe = F.normalize(q_pure_text_emb + valid_img_embs.mean(dim=0, keepdim=True), p=2, dim=-1)
        else:
            q_probe = q_pure_text_emb
            log_full += "*(No images detected, degraded to pure text probe)*\n\n---\n\n"

        log_full += "✨ **[Phase 2: Multimodal Probe Extraction] (Probe -> Text)**\n\n"
        c_text_embs = encode_text_siglip(text_contexts).unsqueeze(0)
        with torch.no_grad():
            text_scores = compressor_model.compute_score(q_probe, c_text_embs).cpu().numpy().flatten()
            
        sel_texts, text_log = apply_top_p_filtering(text_scores, text_contexts, top_p_val)
        log_full += text_log
        log_area.success(log_full)
        compressed_text = "\n\n".join(sel_texts)

        with st.spinner(f"☁️ Calling Cloud VLM [{MODEL_NAME}]..."):
            b64_images_all = [image_to_base64(img) for img in all_images]
            b64_images_valid = [image_to_base64(img) for img in valid_images]
            
            prompt_A = f"You are a top medical expert. Answer the query based on ALL these noisy multidisciplinary notes and uploaded images:\n\n[Raw Noisy Data]:\n{context_text}\n\n[Query]: {query}"
            response_A, pt_A, ct_A = call_cloud_vlm(prompt_A, base64_images=b64_images_all)
            
            prompt_B = f"You are a top medical expert. Answer the query based ONLY on the filtered images and the highly purified clinical evidence extracted by our AI router:\n\n[Purified Core Evidence]:\n{compressed_text}\n\n[Query]: {query}"
            response_B, pt_B, ct_B = call_cloud_vlm(prompt_B, base64_images=b64_images_valid)

        with result_area:
            compression_ratio = ((pt_A - pt_B) / pt_A * 100) if pt_A > 0 else 0
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("❌ Channel A Input Tokens", f"{pt_A:,}")
            m2.metric("✅ Channel B Input Tokens", f"{pt_B:,}", delta=f"-{pt_A - pt_B:,} (Save {compression_ratio:.1f}%)", delta_color="inverse")
            m3.metric("❌ Channel A Output Tokens", f"{ct_A:,}")
            m4.metric("✅ Channel B Output Tokens", f"{ct_B:,}")
            
            st.markdown("### ❌ Channel A: Baseline Cloud VLM (Drowning in Information Tsunami)")
            st.error(response_A)
            st.markdown("### ✅ Channel B: MANN-Engram Edge-Cloud Collab (Top-P Purified)")
            st.success(response_B)

    except Exception as e:
        log_area.error(f"🚨 Crash:\n{traceback.format_exc()}")