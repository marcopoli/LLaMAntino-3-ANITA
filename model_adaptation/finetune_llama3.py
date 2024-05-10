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

max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
model_name = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 8,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 42,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    #modules_to_save = ["embed_tokens"] # add if you want to perturbate embedding layers for a new language adaptation
)

############## ALPACA STYLE INSTRUCT TEMPLATE ##########################
'''
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
def formatting_prompts_func(examples):
    instructions = examples["prompt"]
    inputs       = examples["input"]
    outputs      = examples["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
'''
########## LLAMA 3 TEMPLATE EXAMPLE ###################

llama3_template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA (Advanced Natural-based interaction for the ITAlian language). Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo. <|eot_id|> <|start_header_id|>user<|end_header_id|>

{} <|eot_id|> <|start_header_id|>assistant<|end_header_id|>

{} <|eot_id|> <|end_of_text|>
"""

########## LLAMA 3 CONVERSATION TEMPLATE EXAMPLE ###################
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(example):
    prompt_messages = example["messages"]
    templ = (tokenizer.apply_chat_template(prompt_messages, tokenize=False)+"<|end_of_text|>").replace("<|start_header_id|>assistant<|end_header_id|>\n\n<|end_of_text|>","<|end_of_text|>")
    templ = templ.replace("<|eot_id|>"," <|eot_id|> ").replace("<|begin_of_text|>","<|begin_of_text|> ")
    example["text"] = templ
    return example

from datasets import load_dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split = "train_sft")
dataset = dataset.map(formatting_prompts_func, batched = False,)

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 10,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 8,
        warmup_steps = 100,
        learning_rate = 2e-5, #smaller steps for 2e-5 for unsupervised ITA - standard 2e-4 for finetune
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit",#"adamw_torch", #"paged_adamw_8bit",#paged_adamw_32bit
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
new_model = model_name+"_SFT_adapters"

trainer.model.save_pretrained(new_model)
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
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(new_model+"_final_ultra")
model.save_pretrained(new_model+"_final_ultra", safe_serialization=True)
