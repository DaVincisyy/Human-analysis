import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model

# ================= 1. 配置 =================
MODEL_ID = "openai/clip-vit-base-patch32"
DATASET_PATH = "/home/syy/local_datasets/pedes"
ADAPTER_SAVE_DIR = "./clip_finetuned_adapter"

# ================= 2. 加载模型 =================
print("正在加载基础模型...")
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

# ================= 3. 数据预处理 =================
def process_data(examples):
    # 处理图像和文本，不返回 tensors，让 datasets 自动处理
    inputs = processor(
        text=examples['text'], 
        images=examples['image'], 
        padding="max_length", 
        truncation=True,
        max_length=77
    )
    return inputs

print("正在预处理数据集（这可能需要一点时间）...")
dataset = load_dataset(DATASET_PATH)

train_data = dataset['train'].select(range(5000)).map(
    process_data, 
    batched=True, 
    remove_columns=dataset['train'].column_names
)

# ================= 4. 设置 LoRA =================
config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], 
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# ================= 5. 【核心修复】自定义 Trainer =================
# 因为 CLIPModel 需要 return_loss=True 才会输出 loss
class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): # 关键是加了这个 **kwargs
        if "num_items_in_batch" in kwargs:
            # 如果你好奇这个参数是干嘛的：它是新版用来做更好的梯度缩放的
            pass 
        
        outputs = model(**inputs, return_loss=True)
        return (outputs.loss, outputs) if return_outputs else outputs.loss
# ================= 6. 训练参数 =================
training_args = TrainingArguments(
    output_dir="./clip_results",
    per_device_train_batch_size=16, # 如果显存不足（如低于 12G），请改为 8 或 4
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,
    fp16=True if torch.cuda.is_available() else False, 
    remove_unused_columns=False, 
    report_to="none"
)

# ================= 7. 启动训练 =================
trainer = CLIPTrainer( # 使用我们自定义的 CLIPTrainer
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=default_data_collator,
)

print("开始训练...")
trainer.train()

# 保存最终结果
model.save_pretrained(ADAPTER_SAVE_DIR)
processor.save_pretrained(ADAPTER_SAVE_DIR)
print(f"微调完成！权重已保存至 {ADAPTER_SAVE_DIR}")