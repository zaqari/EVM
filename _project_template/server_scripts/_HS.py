import pandas as pd
import numpy as np
import torch
import os
from datetime import datetime as dt
from tqdm import tqdm
# If on the remote server
# from kgen2.LM.LM.vectorize_data import vectorize_classify_2_models as vectorize
from kgen2.LM.LM.hfclassifier import model
# from kgen2.LM.LM.RoBERTa import RoBERTa

# cut-off point for what counts as a mondo-sized data set to split into subdata.
big_data_set_cutoff = 5000


###########################################################################################
###### Basic set-up
###########################################################################################
print('CUDA:', torch.cuda.is_available())

start = dt.now()
PATH = '/home/zprosen/d/X_Haters/'

dataset = os.path.join(PATH, 'merged-dataset.csv')

output_name = 'HS_index.csv'
OUTPUT_PATH = os.path.join(PATH, output_name)

print(PATH, '\n\n')

level = [7]



###########################################################################################
###### logging
###########################################################################################
# LOG_PATH = os.path.join(PATH, 'submission_id_logging.csv')
# if not os.path.exists(LOG_PATH):
#     log = pd.DataFrame(columns=['submission_id'])
#     log.to_csv(os.path.join(PATH, 'submission_id_logging.csv'), index=False, encoding='utf-8')
#


###########################################################################################
###### Classifier
###########################################################################################
classifier = model(model="facebook/roberta-hate-speech-dynabench-r4-target", device='cuda', special_tokens=True, layers=level)
classifier.tokenizer.add_tokens(['<QUOTE>'], special_tokens=True)
classifier.eval()



###########################################################################################
###### Word Embeddings
###########################################################################################
# wvs = RoBERTa(device='cuda', special_tokens=True, layers=level)
# wvs.tokenizer.add_tokens(['<QUOTE>'], special_tokens=True)
# wvs.eval()



###########################################################################################
###### Process
###########################################################################################
meta_data_cols = ['column', 'url', 'author', 'date']
df = pd.read_csv(dataset)
print(list(df))

if not os.path.exists(OUTPUT_PATH):
    pd.DataFrame(columns=meta_data_cols+['HS']).to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

print('Original Post')
for i in tqdm(df.drop_duplicates(subset=['OP_Original_Url']).index):
    rating = torch.softmax(classifier(df['OP_Full_Text'][i]).detach().cpu(), dim=-1)[-1].item()
    row = {
        'column': 'OP',
        'url': df['OP_Original_Url'][i],
        'author': df['OP_Author'][i],
        'date': df['OP_Date'][i],
        'rating': rating
    }

    pd.DataFrame([row]).to_csv(OUTPUT_PATH, header=False,index=False, encoding='utf-8', mode='a')
print('-+-+-+-+-\n')

print('Replies')
for i in tqdm(df.loc[~df['reply_Full_Text'].isna()].index):
    rating = torch.softmax(classifier(df['reply_Full_Text'][i]).detach().cpu(), dim=-1)[-1].item()
    row = {
        'column': 'reply',
        'url': df['reply_Original_Url'][i],
        'author': df['reply_Author'][i],
        'date': df['reply_Date'][i],
        'rating': rating
    }

    pd.DataFrame([row]).to_csv(OUTPUT_PATH, header=False,index=False, encoding='utf-8', mode='a')


print('=======][=======\n')