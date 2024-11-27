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
output_name = os.path.join(PATH, 'ckpt={}.pt')
dataset = os.path.join(PATH, 'merged-dataset.csv')

print(PATH, '\n\n')

level = [7]


###########################################################################################
###### Basic set-up
###########################################################################################




###########################################################################################
###### Process
###########################################################################################
def convert_HS_string(x):
    output = torch.FloatTensor([float(v) for v in x.split(']]')[0].split('[[')[-1].split(', ')])
    return torch.softmax(3*output, dim=-1)[-1].item()


df = pd.read_csv(dataset)
df['line_no'] = df.index.values
df.index = range(len(df))

meta_data_cols = [col for col in list(df) if col not in ['text', 'SOTU']]


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

for i, year in enumerate(df['YEAR'].unique()):
    print('\n+++++++{}+++++++'.format(year))

    sub = df.loc[df['YEAR'].isin([year])]

    GRAPH.fit(sub['SOTU'].values.tolist(), sub['text'].values.tolist())

    GRAPH.meta_data = sub[meta_data_cols].to_dict(orient='records')


    if ((i+1) % 5) == 0:
        GRAPH.checkpoint(str(output_name).format(i+1))

        GRAPH = ceda_model(
            sigma=1.,
            device='cuda',
            wv_model='roberta-base',
            wv_layers=level
        )

GRAPH.checkpoint(str(output_name).format(i+1))
print('=======][=======\n')
