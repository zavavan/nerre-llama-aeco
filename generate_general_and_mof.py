import os
import sys
import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from pathlib import Path
import json, jsonlines
from datetime import datetime

def main1(
    load_8bit: bool = True,
    base_model: str = '/PATH/TO/MODEL',
    lora_weights: str = '/PATH/TO/FINETUNED/WEIGHTS',
    input_dir: str = "/PATH/TO/TESTDATA/", 
    results_dir: str = "",
    task: str="general",
    fold: int=0,
):
    assert input_dir.endswith('/'), print("input_dir should end with '/'")
    input_dir=input_dir+task
    STOP_TOKEN='\n\nEND\n\n'

    starttime = datetime.strptime(str(datetime.now()),"%Y-%m-%d %H:%M:%S.%f")

    if not os.path.exists(results_dir):
        print("mkdir...",str(results_dir))
        os.makedirs(results_dir)

    val_data=[]
    
    #read jsonl as val_data
    with jsonlines.open(input_dir+f'/fold_{fold}/val.jsonl') as f:
        for line in f:
            val_data.append(line)

    #load model
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map="auto",
        )
    model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map="auto",
        )

    with open(results_dir+f'/run_{fold}.jsonl','w') as outfile:

        for d in tqdm(val_data):
            prompt = d['prompt'][:] # we are looking at original .jsonl file
            model_input = tokenizer(prompt,return_tensors='pt').to('cuda')
            model.eval()
            with torch.no_grad():
                response = tokenizer.decode(model.generate(**model_input,do_sample=False, max_new_tokens=1024)[0], skip_special_tokens=True)
                response = response.replace(prompt,"")
                if response.endswith(STOP_TOKEN):
                    response = response.replace(STOP_TOKEN,"")
            model.train()

            d['gpt3_completion']=response #Note that this is not GPT3 completion but made it consistent for evaluation code
        
            json.dump(d,outfile)
            outfile.write('\n')
    endtime = datetime.strptime(str(datetime.now()),"%Y-%m-%d %H:%M:%S.%f")
    print("Time consumed ",endtime-starttime)

if __name__ == "__main__":
    fire.Fire(main1)
