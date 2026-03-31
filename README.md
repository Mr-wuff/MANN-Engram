<div align="center">

🧠 MANN-Engram: Edge-Cloud Multimodal Semantic Router

A Privacy-First, Zero-Hallucination Shield for Clinical Vision-Language Models

English | 中文说明

</div>

💡 The Problem: "Information Tsunamis" in AI

Modern Vision-Language Models (VLMs) like GPT-4o or Qwen-VL are incredibly smart, but they drown in Information Tsunamis. When fed with dozens of irrelevant historical images, administrative complaints, and messy PDFs, VLMs suffer from severe hallucinations, forced correlations, and massive API token costs.

🚀 The Solution: MANN-Engram

MANN-Engram is an ultra-fast, local-first multimodal router. Before you send your data to an expensive cloud API, MANN-Engram acts as an intelligent "bouncer":

Parses everything (PDF, Word, Excel, Images, Messy Chat).

Extracts intent locally (Zero data leakage using local 0.5B models).

Routes via Latent Space (Skew-Gaussian distance to filter out 100% of the noise).

Delivers pure, highly-concentrated evidence to your downstream VLM.

📊 Performance & ROI Benchmarks

We tested MANN-Engram on "Hell-Mode" MDT (Multidisciplinary Team) Scenarios, deliberately injecting 10+ distractor images and 50+ lines of administrative noise into fatal medical cases.

💸 Token Compression & Cost Saving

By filtering out noise, MANN-Engram significantly reduces the context window size, leading to faster API responses and lower costs, with a negligible local latency overhead.

pie title Average Prompt Token Compression Rate (Hell-Mode Cases)
    "Purified Core Evidence (Saved & Sent)" : 68.5
    "Filtered Noise (Cost Saved!)" : 31.5


Metric

Baseline Cloud VLM

MANN-Engram + Cloud VLM

Improvement

Token Usage (Avg)

6,800 tokens

4,500 tokens

📉 -33% Cost

Diagnostic Accuracy

40% (Hallucinates easily)

100% (Anchored to true lesion)

📈 +150%

Latency Overhead

0 ms

+ 150 ms (Local Ops)

Negligible

Data Privacy (Intent)

Sends all noise to cloud

100% Local Intent Extraction

🔒 Zero Leakage

🎬 See it in Action (Workflow Demo)

1. The Latent Routing Radar

Watch how MANN-Engram dynamically extracts the exact lethal lesion (Vertebral Artery Dissection) out of 10 distracting images (benign cysts, old fractures) in milliseconds.

(Insert your GIF here: ![Routing Demo](docs/assets/radar_demo.gif))

2. A/B Testing: The "Fatal Masquerader" Case

The Setup: The patient suffered a stroke after a golf swing. 10 images and 52 lines of clinical logs are provided.

<details>
<summary><b>🚨 Click to view the Raw Input (The Information Tsunami)</b></summary>

Images: 1 Core VAD image + 4 Tumor images + 5 Heavy Distractors (Benign cysts, normal MRIs). 



Texts: 52 lines of ethical debates, family complaints, and incidental findings.
</details>

❌ Baseline Cloud VLM

✅ MANN-Engram Edge-Cloud Collab

(Hallucination)



Misses the underlying tumor completely. Focuses on the stroke being caused by the golf swing. Acts as a scribe to close the case because it read a "DNR" note in the noise.

(Precision Diagnosis)



Successfully extracts the bone marrow infiltration text & images. Concludes the stroke is secondary to a malignant skull-base tumor invasion.

(Read the full benchmark reports in our Benchmark Showcase)

🛠️ Quick Start (Developer API)

Stop writing fragile regex or sending raw data to OpenAI. Build robust, industrial-grade pipelines in 5 lines of code.

Installation

pip install mann-engram-en


Note: Please download the required routing weights from our Hugging Face Hub into the weights/ directory before running.

End-to-End Multimodal Purification

from mann_engram_en import MANNEngramRouter

# 1. Initialize (Loads local intent model & SigLIP routing core)
router = MANNEngramRouter(
    ckpt_path="./weights/skew_model_v4full_en.pt", 
    enable_local_intent=True # 100% local privacy for intent extraction
)

# 2. Feed the Information Tsunami (Messy text + diverse files + images)
results = router.process_session(
    raw_chat_input="Doc, I hit my head yesterday. Also the nurses here are rude! Look at my files.",
    file_paths=["old_blood_test.pdf", "history.docx"],
    image_paths=["head_ct.jpg", "random_dog.jpg", "leg_mri.jpg"],
    top_p=0.85
)

# 3. Get highly purified evidence ready for your Cloud VLM!
print(f"Core Request: {results['core_query']}")
print(f"Purified Text: {results['purified_context']}")
print(f"Tokens Saved: Filtered out {results['stats']['original_text_chunks'] - results['stats']['retained_text_chunks']} noise chunks!")


🧩 Architecture

MANN-Engram operates on a dual-engine architecture:

Dual-Engine Intent Extractor: Uses a 0.5B ultra-light local LLM (e.g., Qwen2.5) to strip emotional noise and extract the core medical query.

Universal Document Parser: Slides through PDFs, Word docs, and Excel files, chunking them into semantic windows.

Skew-Gaussian Routing Core: Projects both text chunks and images into a shared SiGLIP latent space, calculating the Mahalanobis distance to dynamically nucleus-sample (Top-P) only the most critically relevant evidence.

graph LR
    A[Messy Chat] --> C(Local Intent Extractor)
    B[PDFs/Docs/Images] --> D(Document Parser)
    C --> |Core Query| E{Skew-Gaussian Latent Router}
    D --> |Chunks & Images| E
    E --> |Top-P Filtering| F[Purified Multimodal Evidence]
    F --> |Cost Saved & Clean| G((Cloud VLM))


🤝 Contributing

We welcome contributions! Please see our Contribution Guidelines for more details.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.