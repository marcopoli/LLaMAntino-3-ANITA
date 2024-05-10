<img src="https://cdn-uploads.huggingface.co/production/uploads/5df8bb21da6d0311fd3d540f/cZoZdwQOPdQsnQmDXHcSn.png" alt="llamantino3_anita" border="0" width="800px">
<hr>
<!--<img src="https://i.ibb.co/6mHSRm3/llamantino53.jpg" width="200"/>-->

<p style="text-align:justify;"><b>LLaMAntino-3-ANITA-8B-Inst-DPO-ITA</b> is a model of the <a href="https://huggingface.co/swap-uniba"><b>LLaMAntino</b></a> - <i>Large Language Models family</i>.
The model is an instruction-tuned version of <a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"><b>Meta-Llama-3-8b-instruct</b></a> (a fine-tuned <b>LLaMA 3 model</b>).
This model version aims to be the a <b>Multilingual Model</b> 🏁 -- EN 🇺🇸 + ITA🇮🇹 -- to further fine-tune for the Specific Italian Task</p>


The 🌟**ANITA project**🌟 *(**A**dvanced **N**atural-based interaction for the **ITA**lian language)*
wants to provide Italian NLP researchers with an improved model the for Italian Language 🇮🇹 use cases.

<hr>

## Model Details
*Last Update: 10/05/2024*<br>

<img src="https://github.com/marcopoli/LLaMAntino-3-ANITA/assets/25221576/dd261ecf-88c5-487e-8cd3-cbe7fb7ed1fe" width="200"> <br>[https://huggingface.co/m-polignano-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA](https://huggingface.co/m-polignano-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA) <br>

<hr>

## Specifications

- **Model developers**: <br><a href="https://marcopoli.github.io/">Ph.D. Marco Polignano</a> - University of Bari Aldo Moro, Italy <br> <a href="https://huggingface.co/swap-uniba">SWAP Research Group</a> <br>
- **Variations**: The model release has been **supervised fine-tuning (SFT)** using **QLoRA** 4bit, on two instruction-based datasets. **DPO** approach over the *jondurbin/truthy-dpo-v0.1* dataset is used to align with human preferences for helpfulness and safety.
- **Input**: Models input text only.
- **Language**: Multilingual🏁 + Italian 🇮🇹
- **Output**: Models generate text and code only.
- **Model Architecture**: *Llama 3 architecture*.
- **Context length**: 8K, 8192.
- **Library Used**: [Unsloth](https://unsloth.ai/)
<hr>

## Playground

To use the model directly, there are many ways to get started, choose one of the following ways to experience it.

### Prompt Template
```
<|start_header_id|>system<|end_header_id|>

{ SYS Prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

{ USER Prompt }<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{ ASSIST Prompt }<|eot_id|>
````

### Transformers

For direct use with `transformers`, you can easily get started with the following steps.

- Firstly, you need to install transformers via the command below with `pip`.

  ```bash
  pip install -U transformers
  ```

- Right now, you can start using the model directly.

  ```python
  import torch
  from transformers import (
      AutoModelForCausalLM,
      AutoTokenizer,
  )

  base_model = "m-polignano-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
  model = AutoModelForCausalLM.from_pretrained(
      base_model,
      torch_dtype=torch.bfloat16,
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained(base_model)

  messages = [
      {"role": "system", "content": {"role": "system", "content": "Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA \
      (Advanced Natural-based interaction for the ITAlian language). \
      Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo. "},
      {"role": "user", "content": "Why is the sky blue?"}
  ]

  #Method 1
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
  for k,v in inputs.items():
      inputs[k] = v.cuda()
  outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.6)
  results = tokenizer.batch_decode(outputs)[0]
  print(results)

  #Method 2
  import transformers
  pipe = transformers.pipeline(
      model=model,
      tokenizer=tokenizer,
      return_full_text=False, # langchain expects the full text
      task='text-generation',
      max_new_tokens=512, # max number of tokens to generate in the output
      temperature=0.6,  #temperature for more or less creative answers
      do_sample=True,
      top_p=0.9,
  )

  sequences = pipe(messages)
  for seq in sequences:
      print(f"{seq['generated_text']}")
  
  ```

- Additionally, you can also use a model with **4bit quantization** to reduce the required resources at least. You can start with the code below.

  ```python
  import torch
  from transformers import (
      AutoModelForCausalLM,
      AutoTokenizer,
      BitsAndBytesConfig,
  )

  base_model = "m-polignano-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=False,
  )
  model = AutoModelForCausalLM.from_pretrained(
      base_model,
      quantization_config=bnb_config,
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained(base_model)

  messages = [
     {"role": "system", "content": {"role": "system", "content": "Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA \
     (Advanced Natural-based interaction for the ITAlian language). \
     Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo. "},
      {"role": "user", "content": "Why is the sky blue?"}
  ]

  #Method 1
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
  for k,v in inputs.items():
      inputs[k] = v.cuda()
  outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.6)
  results = tokenizer.batch_decode(outputs)[0]
  print(results)

  #Method 2
  import transformers
  pipe = transformers.pipeline(
      model=model,
      tokenizer=tokenizer,
      return_full_text=False, # langchain expects the full text
      task='text-generation',
      max_new_tokens=512, # max number of tokens to generate in the output
      temperature=0.6,  #temperature for more or less creative answers
      do_sample=True,
      top_p=0.9,
  )

  sequences = pipe(messages)
  for seq in sequences:
      print(f"{seq['generated_text']}")

  ```

### Unsloth

For direct use with `unsloth`, you can easily get started with the following steps.

- Firstly, you need to install unsloth via the command below with `pip`.
  ```bash
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps xformers trl peft accelerate bitsandbytes
  ```

- Initialize and optimize the model before use.
  ```python
  from unsloth import FastLanguageModel
  import torch

  base_model = "m-polignano-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name = base_model,
      max_seq_length = 8192,
      dtype = None,
      load_in_4bit = True, # Change to `False` if you don't want to use 4bit quantization.
  )
  FastLanguageModel.for_inference(model)
  ```

- Right now, you can start using the model directly.
  ```python
  messages = [
      {"role": "system", "content": {"role": "system", "content": "Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA \
     (Advanced Natural-based interaction for the ITAlian language). \
     Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo. "},
      {"role": "user", "content": "Why is the sky blue?"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
  for k,v in inputs.items():
      inputs[k] = v.cuda()
  outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.6)
  results = tokenizer.batch_decode(outputs)[0]
  print(results)
  ```



<hr>

## Unsloth

<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made with unsloth.png" width="200px" align="center" />

[Unsloth](https://unsloth.ai), a great tool that helps us easily develop products, at a lower cost than expected.

## Citation instructions
```bibtex
@misc{basile2023llamantino,
      title={LLaMAntino: LLaMA 2 Models for Effective Text Generation in Italian Language}, 
      author={Pierpaolo Basile and Elio Musacchio and Marco Polignano and Lucia Siciliani and Giuseppe Fiameni and Giovanni Semeraro},
      year={2023},
      eprint={2312.09993},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```bibtex
@article{llama3modelcard,
  title={Llama 3 Model Card},
  author={AI@Meta},
  year={2024},
  url = {https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md}
}
```
