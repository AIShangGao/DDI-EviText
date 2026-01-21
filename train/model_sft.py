import os
from pathlib import Path

# ================= 增加 NCCL 超时设置 =================
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import re
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    trainer_callback,
)
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from liger_kernel.transformers import AutoLigerKernelForCausalLM

import swanlab
from swanlab.plugin.notification import DingTalkCallback
from swanlab.integration.transformers import SwanLabCallback

# ================= SwanLab 登录与回调配置 =================
_swanlab_api_key = os.getenv("SWANLAB_API_KEY")
if not _swanlab_api_key:
    raise EnvironmentError("Missing env SWANLAB_API_KEY")
swanlab.login(api_key=_swanlab_api_key)

_dingtalk_webhook = os.getenv("DINGTALK_WEBHOOK_URL")
_dingtalk_secret = os.getenv("DINGTALK_SECRET")
if not _dingtalk_webhook or not _dingtalk_secret:
    raise EnvironmentError("Missing env DINGTALK_WEBHOOK_URL and/or DINGTALK_SECRET")

dingtalk_callback = DingTalkCallback(
    webhook_url=_dingtalk_webhook,
    secret=_dingtalk_secret,
)

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="Qwen3-LoRA",  # 注意修改
)

# ================= 配置路径（相对路径 + 可用环境变量覆盖）=================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "Qwen3" / "4B_Instruct_2507")))
DATA_BASE_DIR = Path(os.getenv("DATA_BASE_DIR", str(PROJECT_ROOT / "dataset" / "drugbank" / "S1")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "experiments" / "Qwen3" / "checkpoints")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# map 多进程与最大长度也做成可配置（公开项目更友好）
NUM_PROC = int(os.getenv("NUM_PROC", "12"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "16384"))

# ================= 1. 初始化 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================= 2. 数据处理函数 =================
def process_func(example):
    instruction = tokenizer(
        f"<s><|im_start|>system\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ================= 2.1 温和过采样实现（Tempered Oversampling） =================
def _ensure_class_id_column(ds, output_col: str = "output", class_col: str = "class_id"):
    if class_col in ds.column_names:
        return ds

    pattern = re.compile(r"Class:\s*(\d+)")

    def _parse_batch(batch):
        outs = batch.get(output_col, [])
        ids = []
        for s in outs:
            if not isinstance(s, str):
                ids.append(-1)
                continue
            m = pattern.search(s)
            ids.append(int(m.group(1)) if m else -1)
        return {class_col: ids}

    ds = ds.map(_parse_batch, batched=True, num_proc=NUM_PROC)

    # 简单校验：若存在 -1，说明解析失败（通常是 output 模板被改动或数据缺失）
    bad_idx = None
    for i, cid in enumerate(ds[class_col]):
        if int(cid) < 0:
            bad_idx = i
            break
    if bad_idx is not None:
        raise ValueError(
            f"Failed to parse class_id from '{output_col}'. Example output: {ds[bad_idx][output_col]}"
        )

    return ds

def tempered_oversample_dataset(
    ds,
    class_col: str = 'class_id',
    ref_count: int = 1000,
    tau: float = 0.6,
    max_multiplier = 100,
    seed: int = 42,
):
    
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0, 1], got {tau}")

    class_ids = np.asarray(ds[class_col], dtype=np.int64)
    unique, counts = np.unique(class_ids, return_counts=True)
    count_dict = {int(u): int(c) for u, c in zip(unique, counts)}

    if -1 in count_dict:
        raise ValueError("Found class_id = -1 in dataset. class_id parsing likely failed.")

    # 计算每个类的目标样本数
    targets = {}
    for cls, n in count_dict.items():
        if n >= ref_count:
            t = n
        else:
            # 动态扩充（tau<1 时不会统一补到 ref_count）
            t = int(np.ceil(n * (ref_count / n) ** tau))
            if max_multiplier is not None:
                t = min(t, n * int(max_multiplier))
            # 防止浮点误差导致 t 略大于 ref_count（尤其 tau=1 时）
            t = min(t, ref_count)
            t = max(n, t)
        targets[int(cls)] = int(t)

    # 构造每类索引（一次遍历，避免反复 np.where 扫描）
    class_to_indices = {cls: [] for cls in targets.keys()}
    for idx, cid in enumerate(class_ids):
        class_to_indices[int(cid)].append(int(idx))

    rng = np.random.RandomState(seed)
    indices = []
    for cls in sorted(class_to_indices.keys()):
        idxs = class_to_indices[cls]
        indices.extend(idxs)
        need = targets[cls] - len(idxs)
        if need > 0:
            extra = rng.choice(idxs, size=need, replace=True)
            indices.extend(extra.tolist())

    # shuffle，避免同类样本（含复制样本）在数据集中成段出现
    indices = rng.permutation(np.asarray(indices, dtype=np.int64)).tolist()
    ds_os = ds.select(indices)

    print(
        f"[tempered_oversample] num_rows: {len(ds)} -> {len(ds_os)} "
        f"(x{len(ds_os)/len(ds):.3f}); ref_count={ref_count}, tau={tau}, max_multiplier={max_multiplier}"
    )
    return ds_os

def load_and_process_dataset(file_name, apply_tempered_oversample: bool = False):
    file_path = DATA_BASE_DIR / file_name
    if not file_path.exists():
        print(f"Warning: File {file_path} not found. Skipping.")
        return None

    print(f"Loading and processing: {file_path}")
    ds = load_dataset("json", data_files=str(file_path), split="train")

    # --- 温和过采样：仅建议对训练集启用 ---
    if apply_tempered_oversample and ENABLE_TEMPERED_OVERSAMPLE:
        ds = _ensure_class_id_column(ds)
        ds = tempered_oversample_dataset(
            ds,
            class_col="class_id",
            ref_count=TEMPERED_REF_COUNT,
            tau=TEMPERED_TAU,
            max_multiplier=TEMPERED_MAX_MULTIPLIER,
            seed=TEMPERED_RANDOM_SEED,
        )


    return ds.map(process_func, remove_columns=ds.column_names, num_proc=NUM_PROC)

# ================= 3. 加载数据集 =================
# 假设文件名如下，请根据实际情况修改文件名
tokenized_train = load_and_process_dataset("train_sft.json", apply_tempered_oversample=True)
tokenized_val = load_and_process_dataset("valid_sft.json", apply_tempered_oversample=False)
# tokenized_test = load_and_process_dataset('test_full_sft.json')

if tokenized_train is None:
    raise ValueError("Training dataset not found!")

# 打印示例以检查
print("Sample Input IDs:", tokenizer.decode(tokenized_train[0]['input_ids']))

# ================= 4. 加载模型 (FSDP模式) =================
# 注意：使用FSDP时，不能使用 device_map="auto"，FSDP会自动管理设备
print("Loading model for FSDP...")
model = AutoLigerKernelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
)

model.enable_input_require_grads()
model.config.use_cache = False

# ================= 5. LoRA 配置 =================
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, 
    r=32, 
    lora_alpha=64, 
    lora_dropout=0.1
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# ================= 6. 训练参数 (含FSDP与早停) =================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=3, # 设置较大轮数，配合早停使用
    learning_rate=5e-5,
    save_on_each_node=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    bf16=True, # 保持 bf16 开启
    
    # --- 评估与早停配置 ---
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=False,
    metric_for_best_model="loss",
    greater_is_better=False
)

# 自定义 Callback：在每个 epoch 结束时保存模型
class SaveAtEpochEndCallback(trainer_callback.TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # 复制一份 control，避免就地修改带来副作用
        # control = control.copy()
        # 告诉 Trainer：在这个 epoch 结束时执行一次 save
        control.should_save = True
        return control

# ================= 7. 初始化 Trainer =================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val, # 传入验证集
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=10), # 验证集loss连续10次不下降则停止
        # swanlab_callback,
        SaveAtEpochEndCallback()
        ], 
)

# ================= 8. 关键修改：应用 PEFT+FSDP 自动包裹策略 =================
# 这一步非常重要，确保 LoRA 层被正确地 shard
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    print("Applying PEFT FSDP auto wrap policy...")
    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    # 将当前模型的包裹策略设置为 PEFT 专用的策略
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

# ================= 9. 开始训练 =================
# 检查是否存在checkpoint以决定是否断点续训
resume = False
if os.path.isdir(OUTPUT_DIR):
    # 检查是否存在以 'checkpoint' 开头的文件夹
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint") and os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    if len(checkpoints) > 0:
        print("Found existing checkpoints, attempting to resume...")
        resume = True
trainer.train(resume_from_checkpoint=resume)