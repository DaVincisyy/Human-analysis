# 🚀 Human-Analysis: Multi-Modal Pedestrian Retrieval & Re-ID

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLO-v8/v11-green.svg" alt="YOLO">
  <img src="https://img.shields.io/badge/Gradio-UI-orange.svg" alt="Gradio">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

An intelligent surveillance analysis platform that combines **Computer Vision (CV)** and **Natural Language Processing (NLP)** to perform real-time person tracking and semantic-based target retrieval. This project features a fine-tuned **CLIP** model using **LoRA** (Low-Rank Adaptation) to achieve high-precision Person Re-Identification (Re-ID).

---

## 📺 Demo UI
> **Note**: Add a screenshot of your Gradio interface here to make it look professional!
> `![UI Screenshot](your_image_url_here)`

---

## ✨ Key Features

- [x] **Deep Multi-Object Tracking (MOT)**: Utilizes **YOLO** with **BoT-SORT** for robust, real-time pedestrian detection.
- [x] **Semantic Search**: Natural language queries like *"A person wearing a red jacket"* to find targets.
- [x] **LoRA Fine-tuning**: Custom training pipeline to refine **OpenAI CLIP** on 230k+ pedestrian samples.
- [x] **Voice-Activated**: Integrated **OpenAI Whisper** for ASR and **Edge-TTS** for voice reports.
- [x] **AI Reasoning**: Powered by **Qwen-Turbo** for intelligent query parsing.

---

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Detection/Tracking** | YOLOv8/v11, BoT-SORT |
| **Visual-Language** | OpenAI CLIP (ViT-B/32) |
| **Model Optimization** | **LoRA (PEFT)** |
| **Speech (ASR/TTS)** | Whisper & Edge-TTS |
| **LLM Reasoning** | Alibaba Qwen-Turbo |
| **Frontend UI** | Gradio |

---

## 🧠 System Architecture

```mermaid
graph TD
    A[Video Input] --> B[YOLO + BoT-SORT Tracking]
    B --> C[ID Gallery / Crops Extraction]
    D[Voice/Text Query] --> E[Qwen-Turbo Parsing]
    E --> F[CLIP + LoRA Semantic Matching]
    C --> F
    F --> G[Gradio Results / TTS Report]
🏋️ LoRA Fine-tuning Details
The model is optimized using Low-Rank Adaptation (LoRA), significantly enhancing its ability to recognize pedestrian attributes while keeping the model lightweight.
Parameter	Value
Rank (r)	16
Alpha	32
Trainable Params	~1.28% of total
Target Modules	q_proj, v_proj, k_proj, out_proj
🔧 Installation & Setup
Clone & Install Dependencies:
code
Bash
git clone https://github.com/DaVincisyy/Human-analysis.git
cd Human-analysis
pip install torch transformers peft ultralytics gradio whisper openai edge-tts
Configure API:
Add your QWEN_API_KEY in humananalysis.py.
Run Application:
code
Bash
python humananalysis.py
👤 Author
DaVincisyy
🎓 Junior Student in Information Engineering
🔭 Focusing on Multi-modal Learning & AI System Integration
📫 Reach out via GitHub Issues!
