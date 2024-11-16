import pandas as pd
import torch
import os
from datetime import datetime as dt
from tqdm import tqdm
# If on the remote server
# from kgen2.mutual_information.end_to_end_analysis import analyzer
# from kgen2.mutual_information.fastGraph import fastGraphWithAnalyzer as FGA
# from kgen2.mutual_information.entropy import entropy_cdf
# from kgen2.LM.LM.RoBERTa import RoBERTa
from kgen2.CEDA import ceda_model



###########################################################################################
###### Basic set-up
###########################################################################################
print('CUDA:', torch.cuda.is_available())

start = dt.now()
PATH = '/home/zprosen/d/X_Haters/'
output_name = os.path.join(PATH, 'ckpt.pt')
dataset = os.path.join(PATH, 'merged-dataset.csv')

print(PATH, '\n\n')

level = [7, -1]



###########################################################################################
###### Process
###########################################################################################
def convert_HS_string(x):
    output = torch.FloatTensor([float(v) for v in x.split(']]')[0].split('[[')[-1].split(', ')])
    return torch.softmax(3*output, dim=-1)[-1].item()


df = pd.read_csv(dataset)
df['line_no'] = df.index.values
df = df.sort_values(by=['OP_Full_Text', 'reply_Full_Text'])
df.index = range(len(df))

meta_data_cols = [col for col in list(df) if ('Full_Text' not in col)]


df = df.loc[~df['OP_Full_Text'].isna()]
df = df.loc[(~df['reply_Full_Text'].isna())]


# print(df.isna().sum(), '\n\n')
#
# df['OP_HS'] = [convert_HS_string(x) for x in tqdm(df['OP_HS'].values)]
# df['HS'] = [convert_HS_string(x) for x in tqdm(df['HS'].values)]

GRAPH = ceda_model(
    sigma=1.,
    device='cuda',
    wv_model='roberta-base',
    wv_layers=level
)

GRAPH.fit(df['OP_Full_Text'].values.tolist(), df['reply_Full_Text'].values.tolist())

# try:
#     GRAPH.checkpoint(output_name)
# except Exception:
#     0

# update_x_col_names = {col:col.replace(' ', '_') for col in ['OP_Author', 'OP_Date', 'OP_Twitter Retweets', 'OP_Twitter Likes', 'OP_Sentiment', 'OP_HS', 'OP_about_jewish_people', 'OP_about_muslim_people', 'OP_Original Url']}
# update_y_col_names = {col:'reply_'+col.replace(' ', '_') for col in ['Author', 'Twitter Retweets', 'Twitter Likes', 'Sentiment', 'HS', 'about_jewish_people', 'about_muslim_people', 'Original Url']}
# df = df.rename(columns=update_x_col_names)
# df = df.rename(columns=update_y_col_names)

GRAPH.meta_data = df[meta_data_cols].to_dict(orient='records')
GRAPH.checkpoint(output_name)

print('=======][=======\n')
