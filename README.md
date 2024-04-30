---
license: llama3
license_name: llama3
license_link: LICENSE
---
<img src="https://cdn-uploads.huggingface.co/production/uploads/5df8bb21da6d0311fd3d540f/cZoZdwQOPdQsnQmDXHcSn.png" alt="llamantino3_anita" border="0" width="800px">
<hr>
<!--<img src="https://i.ibb.co/6mHSRm3/llamantino53.jpg" width="200"/>-->

## Model Details
*Last Update: 29/04/2024*<br>
**LLaMAntino-3-ANITA-8B-sft-ORPO** is a model of the [**LLaMAntino**](https://huggingface.co/swap-uniba) - *Large Language Models family*.
The model is an instruction-tuned version of [**Meta-Llama-3-8b-instruct**](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) (a fine-tuned **LLaMA 3 model**).
This model version aims to be the **Multilingual Base-Model** 🏁 to further fine-tune in the Italian environment. 


The 🌟**ANITA project**🌟 *(**A**dvanced **N**atural-based interaction for the **ITA**lian language)*
wants to provide Italian NLP researchers with an improved model the for Italian Language 🇮🇹 use cases.

<hr>

## Specifications

- **Model developers**: Ph.D. Marco Polignano - University of Bari Aldo Moro, Italy
- **Variations**: The model release has been **supervised fine-tuning (SFT)** using **QLoRA**, on a long list of instruction-based datasets. **DPO** approach over the *HuggingFaceH4/ultrafeedback_binarized* dataset is used to align with human preferences for helpfulness and safety.
- **Input**: Models input text only.
- **Output**: Models generate text and code only.
- **Model Architecture**: *Llama 3 architecture*.
- **Context length**: 8K, 8192.
<hr>

## Playground

To use the model directly, there are many ways to get started, choose one of the following ways to experience it.


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

  base_model = "m-polignano-uniba/LLaMAntino-3-ANITA-8B-sft-DPO"
  model = AutoModelForCausalLM.from_pretrained(
      base_model,
      torch_dtype=torch.bfloat16,
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained(base_model)

  messages = [
      {"role": "system", "content": "Answer clearly and detailed."},
      {"role": "user", "content": "Why is the sky blue ?"}
  ]

  #Method 1
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
  for k,v in inputs.items():
      inputs[k] = v.cuda()
  outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.85, temperature=0.7)
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
      temperature=0.7,  #temperature for more or less creative answers
      do_sample=True,
      top_p=0.85,
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

  base_model = "m-polignano-uniba/LLaMAntino-3-ANITA-8B-sft-DPO"
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
      {"role": "system", "content": "Answer clearly and detailed."},
      {"role": "user", "content": "Why is the sky blue ?"}
  ]

  #Method 1
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
  for k,v in inputs.items():
      inputs[k] = v.cuda()
  outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.85, temperature=0.7)
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
      temperature=0.7,  #temperature for more or less creative answers
      do_sample=True,
      top_p=0.85,
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

  base_model = "m-polignano-uniba/LLaMAntino-3-ANITA-8B-sft-DPO"
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
      {"role": "system", "content": "Answer clearly and detailed."},
      {"role": "user", "content": "Why is the sky blue ?"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
  for k,v in inputs.items():
      inputs[k] = v.cuda()
  outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.85, temperature=0.7)
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

