---
datasets:
- yahma/alpaca-cleaned
- fuliucansheng/InstructionWild
- Crystalcareai/Natural-Instructions-Small-Alpaca
- codefuse-ai/Evol-instruction-66k
- teknium/GPTeacher-General-Instruct
- Chat-Error/wizard_alpaca_dolly_orca
- microsoft/orca-math-word-problems-200k
- qwedsacf/grade-school-math-instructions
- andersonbcdefg/supernatural-instructions-2m
- HuggingFaceH4/ultrachat_200k
- HuggingFaceH4/ultrafeedback_binarized
language:
- en
- it
metrics:
- accuracy
pipeline_tag: text-generation
tags:
- facebook
- meta
- pythorch
- llama
- llama-3
- llamantino
license: other
license_name: llama3
license_link: LICENSE
extra_gated_prompt: >-
  ### META LLAMA 3 COMMUNITY LICENSE AGREEMENT

  Meta Llama 3 Version Release Date: April 18, 2024
  
  "Agreement" means the terms and conditions for use, reproduction, distribution and modification of the
  Llama Materials set forth herein.

  "Documentation" means the specifications, manuals and documentation accompanying Meta Llama 3
  distributed by Meta at https://llama.meta.com/get-started/.

  "Licensee" or "you" means you, or your employer or any other person or entity (if you are entering into
  this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or
  regulations to provide legal consent and that has legal authority to bind your employer or such other
  person or entity if you are entering in this Agreement on their behalf.

  "Meta Llama 3" means the foundational large language models and software and algorithms, including
  machine-learning model code, trained model weights, inference-enabling code, training-enabling code,
  fine-tuning enabling code and other elements of the foregoing distributed by Meta at
  https://llama.meta.com/llama-downloads.

  "Llama Materials" means, collectively, Meta’s proprietary Meta Llama 3 and Documentation (and any
  portion thereof) made available under this Agreement.

  "Meta" or "we" means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your
  principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located
  outside of the EEA or Switzerland).
     
  1. License Rights and Redistribution.

  a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free
  limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Llama
  Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the
  Llama Materials.

  b. Redistribution and Use.

  i. If you distribute or make available the Llama Materials (or any derivative works
  thereof), or a product or service that uses any of them, including another AI model, you shall (A) provide
  a copy of this Agreement with any such Llama Materials; and (B) prominently display “Built with Meta
  Llama 3” on a related website, user interface, blogpost, about page, or product documentation. If you
  use the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is
  distributed or made available, you shall also include “Llama 3” at the beginning of any such AI model
  name.

  ii. If you receive Llama Materials, or any derivative works thereof, from a Licensee as part 
  of an integrated end user product, then Section 2 of this Agreement will not apply to you.

  iii. You must retain in all copies of the Llama Materials that you distribute the following
  attribution notice within a “Notice” text file distributed as a part of such copies: “Meta Llama 3 is
  licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights
  Reserved.”

  iv. Your use of the Llama Materials must comply with applicable laws and regulations
  (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Llama
  Materials (available at https://llama.meta.com/llama3/use-policy), which is hereby incorporated by
  reference into this Agreement.

  v. You will not use the Llama Materials or any output or results of the Llama Materials to
  improve any other large language model (excluding Meta Llama 3 or derivative works thereof).

  2. Additional Commercial Terms. If, on the Meta Llama 3 version release date, the monthly active users
  of the products or services made available by or for Licensee, or Licensee’s affiliates, is greater than 700
  million monthly active users in the preceding calendar month, you must request a license from Meta,
  which Meta may grant to you in its sole discretion, and you are not authorized to exercise any of the
  rights under this Agreement unless or until Meta otherwise expressly grants you such rights.

  3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY
  OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF
  ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED,
  INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT,
  MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR
  DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE LLAMA MATERIALS AND
  ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE LLAMA MATERIALS AND ANY OUTPUT AND
  RESULTS.

  4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING
  OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL,
  INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED
  OF THE POSSIBILITY OF ANY OF THE FOREGOING.

  5. Intellectual Property.

  a. No trademark licenses are granted under this Agreement, and in connection with the Llama
  Materials, neither Meta nor Licensee may use any name or mark owned by or associated with the other
  or any of its affiliates, except as required for reasonable and customary use in describing and
  redistributing the Llama Materials or as set forth in this Section 5(a). Meta hereby grants you a license to
  use “Llama 3” (the “Mark”) solely as required to comply with the last sentence of Section 1.b.i. You will
  comply with Meta’s brand guidelines (currently accessible at
  https://about.meta.com/brand/resources/meta/company-brand/ ). All goodwill arising out of your use
  of the Mark will inure to the benefit of Meta.

  b. Subject to Meta’s ownership of Llama Materials and derivatives made by or for Meta, with
  respect to any derivative works and modifications of the Llama Materials that are made by you, as
  between you and Meta, you are and will be the owner of such derivative works and modifications.

  c. If you institute litigation or other proceedings against Meta or any entity (including a
  cross-claim or counterclaim in a lawsuit) alleging that the Llama Materials or Meta Llama 3 outputs or
  results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other
  rights owned or licensable by you, then any licenses granted to you under this Agreement shall
  terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold
  harmless Meta from and against any claim by any third party arising out of or related to your use or
  distribution of the Llama Materials.

  6. Term and Termination. The term of this Agreement will commence upon your acceptance of this
  Agreement or access to the Llama Materials and will continue in full force and effect until terminated in
  accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in
  breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete
  and cease use of the Llama Materials. Sections 3, 4 and 7 shall survive the termination of this
  Agreement.

  7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of
  the State of California without regard to choice of law principles, and the UN Convention on Contracts
  for the International Sale of Goods does not apply to this Agreement. The courts of California shall have
  exclusive jurisdiction of any dispute arising out of this Agreement.

  ### Meta Llama 3 Acceptable Use Policy

  Meta is committed to promoting safe and fair use of its tools and features, including Meta Llama 3. If you
  access or use Meta Llama 3, you agree to this Acceptable Use Policy (“Policy”). The most recent copy of
  this policy can be found at [https://llama.meta.com/llama3/use-policy](https://llama.meta.com/llama3/use-policy)

  #### Prohibited Uses

  We want everyone to use Meta Llama 3 safely and responsibly. You agree you will not use, or allow
  others to use, Meta Llama 3 to:
  1. Violate the law or others’ rights, including to:
      1. Engage in, promote, generate, contribute to, encourage, plan, incite, or further illegal or unlawful activity or content, such as:
          1. Violence or terrorism
          2. Exploitation or harm to children, including the solicitation, creation, acquisition, or dissemination of child exploitative content or failure to report Child Sexual Abuse Material
          3. Human trafficking, exploitation, and sexual violence
          4. The illegal distribution of information or materials to minors, including obscene materials, or failure to employ legally required age-gating in connection with such information or materials.
          5. Sexual solicitation
          6. Any other criminal activity
      2. Engage in, promote, incite, or facilitate the harassment, abuse, threatening, or bullying of individuals or groups of individuals
      3. Engage in, promote, incite, or facilitate discrimination or other unlawful or harmful conduct in the provision of employment, employment benefits, credit, housing, other economic benefits, or other essential goods and services
      4. Engage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or related professional practices
      5. Collect, process, disclose, generate, or infer health, demographic, or other sensitive personal or private information about individuals without rights and consents required by applicable laws
      6. Engage in or facilitate any action or generate any content that infringes, misappropriates, or otherwise violates any third-party rights, including the outputs or results of any products or services using the Llama Materials
      7. Create, generate, or facilitate the creation of malicious code, malware, computer viruses or do anything else that could disable, overburden, interfere with or impair the proper working, integrity, operation or appearance of a website or computer system
  2. Engage in, promote, incite, facilitate, or assist in the planning or development of activities that present a risk of death or bodily harm to individuals, including use of Meta Llama 3 related to the following:
      1. Military, warfare, nuclear industries or applications, espionage, use for materials or activities that are subject to the International Traffic Arms Regulations (ITAR) maintained by the United States Department of State
      2. Guns and illegal weapons (including weapon development)
      3. Illegal drugs and regulated/controlled substances
      4. Operation of critical infrastructure, transportation technologies, or heavy machinery
      5. Self-harm or harm to others, including suicide, cutting, and eating disorders
      6. Any content intended to incite or promote violence, abuse, or any infliction of bodily harm to an individual
  3. Intentionally deceive or mislead others, including use of Meta Llama 3 related to the following:
      1. Generating, promoting, or furthering fraud or the creation or promotion of disinformation
      2. Generating, promoting, or furthering defamatory content, including the creation of defamatory statements, images, or other content
      3. Generating, promoting, or further distributing spam
      4. Impersonating another individual without consent, authorization, or legal right
      5. Representing that the use of Meta Llama 3 or outputs are human-generated
      6. Generating or facilitating false online engagement, including fake reviews and other means of fake online engagement
  4. Fail to appropriately disclose to end users any known dangers of your AI system
  
  Please report any violation of this Policy, software “bug,” or other problems that could lead to a violation
  of this Policy through one of the following means:
      * Reporting issues with the model: [https://github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)
      * Reporting risky content generated by the model:
      developers.facebook.com/llama_output_feedback
      * Reporting bugs and security concerns: facebook.com/whitehat/info
      * Reporting violations of the Acceptable Use Policy or unlicensed uses of Meta Llama 3: LlamaUseReport@meta.com
extra_gated_fields:
  First Name: text
  Last Name: text
  Date of birth: date_picker
  Country: country
  Affiliation: text
  geo: ip_location  
  By clicking Submit below I accept the terms of the license and acknowledge that the information I provide will be collected stored processed and shared in accordance with the Meta Privacy Policy: checkbox
extra_gated_description: The information you provide will be collected, stored, processed and shared in accordance with the [Meta Privacy Policy](https://www.facebook.com/privacy/policy/).
extra_gated_button_content: Submit
widget:
  - example_title: Hello
    messages:
    - role: user
      content: Hey my name is Julien! How are you?
  - example_title: Winter holidays
    messages:
    - role: system
      content: You are a helpful and honest assistant. Please, respond concisely and truthfully.
    - role: user
      content: Can you recommend a good destination for Winter holidays?
  - example_title: Programming assistant
    messages:
    - role: system
      content: You are a helpful and honest code and programming assistant. Please, respond concisely and truthfully.
    - role: user
      content: Write a function that computes the nth fibonacci number.
inference:
  parameters:
    max_new_tokens: 300
    stop:
    - <|end_of_text|>
    - <|eot_id|>
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

