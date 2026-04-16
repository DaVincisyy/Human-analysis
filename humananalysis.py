import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import gradio as gr
import cv2
import os
import tempfile
import numpy as np
import torch
import whisper
import asyncio
import edge_tts
import re
from PIL import Image
from ultralytics import YOLO
from openai import OpenAI

# 用于加载微调后的 LoRA 模型
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel

# ================= 1. 配置与模型初始化 =================
QWEN_API_KEY = "sk-xxxx"
MODEL_PATH = "yolo26n.pt"
LORA_ADAPTER_PATH = "/home/syy/workspace/project/clip_finetuned_adapter"
BASE_CLIP_MODEL = "openai/clip-vit-base-patch32"

print("正在启动微调版多人ReID语义检索系统...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1.1 加载 YOLO 和 Whisper
yolo_model = YOLO(MODEL_PATH)
asr_model = whisper.load_model("tiny")
client = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 1.2 加载微调后的 CLIP 模型 (关键修改)
print(f"正在加载 LoRA 微调权重: {LORA_ADAPTER_PATH}...")
try:
    # 加载基础模型
    base_clip = CLIPModel.from_pretrained(BASE_CLIP_MODEL).to(device)
    # 叠加 LoRA 适配器
    clip_model = PeftModel.from_pretrained(base_clip, LORA_ADAPTER_PATH).to(device)
    clip_model.eval()
    # 加载配套的处理器
    clip_processor = CLIPProcessor.from_pretrained(BASE_CLIP_MODEL)
    print("微调模型加载成功！")
except Exception as e:
    print(f"加载微调模型失败，请确认训练已完成并保存。错误: {e}")
    # 备选方案：如果加载失败，可以考虑退回到原始模型，或者直接报错
    exit()

# 全局数据结构
id_gallery = {}
video_metadata = {"max_people": 0, "processed": False}

# ================= 2. 核心算法工具 =================

def clean_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

async def tts_speak(text):
    output_path = os.path.join(tempfile.gettempdir(), f"voice_{os.urandom(3).hex()}.mp3")
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    await communicate.save(output_path)
    return output_path

# ================= 3. 视频处理流水线 =================

def process_video_advanced(video_path):
    global id_gallery
    id_gallery = {} 
    
    if not video_path: return None, "请上传视频"
    
    cap = cv2.VideoCapture(video_path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    output_path = os.path.join(tempfile.gettempdir(), "high_stable_track.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    max_on_screen = 0
    
    print("正在执行追踪...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            max_on_screen = max(max_on_screen, len(ids))

            for i, track_id in enumerate(ids):
                conf = confs[i]
                if track_id not in id_gallery or conf > id_gallery[track_id]["conf"]:
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    pad = 10
                    crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
                    if crop.size > 0:
                        id_gallery[track_id] = {
                            "crop": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                            "conf": conf,
                            "id": track_id
                        }
    cap.release()
    writer.release()
    video_metadata.update({"max_people": max_on_screen, "processed": True})
    return output_path, f"分析完成！识别出 {len(id_gallery)} 个唯一身份目标。"

# ================= 4. 语义检索 =================

def handle_ai_search(audio_path, text_input):
    if not video_metadata["processed"]: return "请先分析视频。", None, None
    
    query = text_input
    if audio_path:
        query = asr_model.transcribe(audio_path, language='zh')["text"]
    if not query: return "无效指令", None, None

    match_results = []
    
    try:
        # 1. 语义特征提取 (LLM) - 让标签更符合 ReID
        res = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "system", "content": "You are a ReID assistant. Extract English visual tags like gender, clothing color, accessories from the query."},
                      {"role": "user", "content": f"Query: {query}"}]
        )
        en_query = clean_ascii(res.choices[0].message.content)

        # 2. CLIP 检索 (使用 Transformers 方式)
        all_ids = list(id_gallery.keys())
        all_crops = [id_gallery[tid]["crop"] for tid in all_ids]
        
        if all_crops:
            # 使用微调后的 processor 处理所有图片和文字
            # 将 numpy 数组转为 PIL Image
            pil_crops = [Image.fromarray(c) for c in all_crops]
            
            inputs = clip_processor(
                text=[f"a photo of a person {en_query}"],
                images=pil_crops,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = clip_model(**inputs)
                # 获取相似度概率
                logits_per_image = outputs.logits_per_image # [N, 1]
                probs = logits_per_image.softmax(dim=0).flatten().cpu().numpy()
            
            # 排序
            sorted_indices = np.argsort(probs)[::-1]
            for idx in sorted_indices:
                # 微调后匹配度会更集中，这里阈值可以设低一点或根据效果调整
                if probs[idx] > 0.01: 
                    match_results.append(all_crops[idx])
                if len(match_results) >= 8: break

        # 3. AI 报告
        ai_msg = f"用户搜索：'{query}'。我们在监控中找到了{len(match_results)}个高匹配度个体。请总结。"
        resp = client.chat.completions.create(model="qwen-turbo", messages=[{"role": "user", "content": ai_msg}])
        ai_reply = resp.choices[0].message.content
        
        # 4. 语音
        voice = asyncio.run(tts_speak(ai_reply))
        
        return ai_reply, voice, match_results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"检索失败: {str(e)}", None, None

# ================= 5. UI 界面  =================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 智能AI语义检索系统 (LoRA微调版)")
    
    with gr.Row():
        with gr.Column():
            v_in = gr.Video(label="上传监控视频")
            btn_scan = gr.Button("第一步：启动深度追踪分析", variant="primary")
            v_out = gr.Video(label="追踪结果")
            info = gr.Textbox(label="系统状态")
            
        with gr.Column():
            gr.Markdown("### 🔍 第二步：语义目标检索")
            a_in = gr.Audio(label="语音指令", type="filepath")
            t_in = gr.Textbox(label="文字指令", placeholder="例如：穿红色衣服背黑包的人")
            btn_search = gr.Button("开始语义匹配", variant="secondary")
            
            gr.Markdown("### 📊 检索结果")
            rep_text = gr.Textbox(label="AI 报告")
            rep_voice = gr.Audio(label="播报", autoplay=True)
            out_gal = gr.Gallery(label="匹配到的目标", columns=4)

    btn_scan.click(process_video_advanced, inputs=v_in, outputs=[v_out, info])
    btn_search.click(handle_ai_search, inputs=[a_in, t_in], outputs=[rep_text, rep_voice, out_gal])

if __name__ == "__main__":
    demo.launch()
