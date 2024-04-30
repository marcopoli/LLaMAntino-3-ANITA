import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from unsloth import FastLanguageModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel

### LOAD THE MODEL ###
max_seq_length = 8192
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model_name = "Meta-Llama-3-8B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

### DPO dataset PREP ###
def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})
        return

def apply_dpo_template(example):
    if all(k in example.keys() for k in ("chosen", "rejected")):
        prompt_messages = example["chosen"][:-1]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]
        maybe_insert_system_message(prompt_messages, tokenizer)
        example["text_prompt"] = (tokenizer.apply_chat_template(prompt_messages, tokenize=False)+"<|end_of_text|>").replace("<|start_header_id|>assistant<|end_header_id|>\n\n<|end_of_text|>","<|end_of_text|>")
        example["text_chosen"] = (tokenizer.apply_chat_template(chosen_messages, tokenize=False)+"<|end_of_text|>").replace("<|start_header_id|>assistant<|end_header_id|>\n\n<|end_of_text|>","<|end_of_text|>")
        example["text_rejected"] = (tokenizer.apply_chat_template(rejected_messages, tokenize=False)+"<|end_of_text|>").replace("<|start_header_id|>assistant<|end_header_id|>\n\n<|end_of_text|>","<|end_of_text|>")
    return example

from datasets import load_dataset
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split = "train_prefs")
dataset = dataset.map(apply_dpo_template, batched = False,)
dataset = dataset.remove_columns(["chosen", "rejected","prompt","messages","score_chosen","score_rejected"])
dataset = dataset.rename_column("text_prompt", "prompt")
dataset = dataset.rename_column("text_chosen", "chosen")
dataset = dataset.rename_column("text_rejected", "rejected")
dataset = dataset.train_test_split(test_size=0.01)

### DPO TRAINER ###
#DPO model trainer- One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer
PatchDPOTrainer()

from transformers import TrainingArguments
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 12,
        warmup_ratio = 0.1,
        num_train_epochs = 2,
        learning_rate = 5e-6,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
    ),
    beta = 0.1,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)
dpo_trainer.train()

### SAVE the NEW MODEL ###
new_model = model_name+"_adapters"
dpo_trainer.save_model(new_model)

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

tokenizer.save_pretrained(new_model+"_final")
model.save_pretrained(new_model+"_final", safe_serialization=True)
