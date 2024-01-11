from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import configparser
import transformers
import argparse
import torch
import glob

import pandas as pd

from tqdm import tqdm


def create_prompt(language, text):
    with open('data/context.txt', 'r') as f:
        context = f.read()

    with open('data/guidelines.txt', 'r') as f:
        guidelines = f.read()
        
    with open('data/output.txt', 'r') as f:
        output = f.read()
        
    with open('data/example.txt', 'r') as f:
        example = f.read()
        
    with open('data/question.txt', 'r') as f:
        question = f.read()
        
    context = context.replace('{language}', language)
    question = question.replace('{text}', text)
    
    prompt = context + guidelines + output + example + question
    
    return prompt


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("conf.env")
    hf_loging_token = config["HF"]["HUGGINGFACE_LOGIN_TOKEN"]
    
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        rope_scaling={"type": "dynamic", "factor": 2.0}
    ).to('cuda')
    
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    
    lang_code2language = {
        'eng': 'English',
        'spa': 'Spanish',
        'fra': 'French',
        'ita': 'Italian',
        'deu': 'German'
    }
    
    for csv_path in sorted(glob.glob('data/*.csv')):
        df = pd.read_csv(csv_path)
        lang_code = csv_path.split('/')[-1].split('_')[0]
    
        for i, row in df.iterrows():
            prompt = create_prompt(lang_code2language[lang_code], row.clean_text)
        
            # sequences = pipeline(
            #     prompt[:1000],
            #     do_sample=True,
            #     top_k=10,
            #     num_return_sequences=1,
            #     eos_token_id=tokenizer.eos_token_id,
            #     max_length=1000,
            # )
            
            inputs = tokenizer(prompt[:1000], return_tensors="pt").to('cuda')
            gen_out = model.generate(**inputs, max_new_tokens=1000)
            
            answer = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            df.at[i, 'answer'] = answer

            # answer = ''
            # for seq in sequences:
            #     answer += seq['generated_text']
                
            # df.at[i, 'answer'] = seq['generated_text']
            break
        break
    
    df.to_csv(f'output/{lang_code}.csv', index=False)