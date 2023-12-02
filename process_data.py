import json, jsonlines
import sys


NERRErepo_dir = '../NERRE/'

if NERRErepo_dir.endswith('/'):
    pass
else:
    NERRErepo_dir+='/'
read=[]
valsize=0

# process doping data for llama-2 finetuning
for mode in ['json','engextra','eng']:
    with jsonlines.open(NERRErepo_dir+'doping/data/'+f'training_{mode}.jsonl') as f:
        for line in f:#.iter():
            read.append(line)
    write=[{'input':dat['prompt'],'output':dat['completion']} for dat in read]
    # Note that doping 'prompt' has instruction in it, feel free to explore without it or with different ones.


    train_dat,val_dat = write[valsize:], write[:valsize] #valset can be used needed for llama-2 finetuning but we will give it blank list here
    for ftmode in ['train','val']:
        if ftmode=='train':
            data = train_dat[:]
        else:
            data = val_dat[:]
        with open(NERRErepo_dir+'doping/data/'+f"doping_data_forllama_{mode}scheme_{ftmode}.json","w") as file1:
            json.dump(data,file1)

# process general and MOF data for llama-2 finetuning
for taskdir in ['experiments_general','experiments_mof_gpt3']:
    for i in range(5):
        folddir = f'fold_{i}/'
        path = NERRErepo_dir+'general_and_mofs/data/'+taskdir+'/'+folddir
        for mode in ['train','val']: # Note val.jsonl is used as test data
            read=[]
            with jsonlines.open(path+f'{mode}.jsonl') as f:
                read = [line for line in f]
            write=[{'input':dat['prompt'],'output':dat['completion']} for dat in read]
            # Feel free to test other instructions or formats. Note that we do not use instruction for this task.


            with open(path+f"{mode}.json","w") as file1: #the processed file is .json file not .jsonl
                json.dump(write,file1)
