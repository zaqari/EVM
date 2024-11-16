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
PATH = '/home/zprosen/d/...'

### CREATE OUTPUT PATH FOLDER
OUTPUT_PATH = os.path.join(PATH, 'ckpt')
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
output_name = os.path.join(OUTPUT_PATH, '{}-ckpt.pt')

### CREATE PROGRESS TRACKER
TRACKER_NAME = os.path.join(PATH, 'tracker.csv')
if not os.path.exists(TRACKER_NAME):
    tracker = pd.DataFrame([{'line_no': -1}])
    tracker.to_csv(TRACKER_NAME, index=False, encoding='utf-8')
    start_at = 0
else:
    start_at = pd.read_csv(TRACKER_NAME)['line_no'].values[-1]

### REMAINING VARIABLES
dataset = os.path.join(PATH, 'merged-dataset.csv')

text_col = 'body'
level = [7, -1]
save_at = 100
print('{}\n {} | {} | {}\n\n'.format(PATH, save_at, text_col, level))



###########################################################################################
###### Process
###########################################################################################
def convert_HS_string(x):
    output = torch.FloatTensor([float(v) for v in x.split(']]')[0].split('[[')[-1].split(', ')])
    return torch.softmax(3*output, dim=-1)[-1].item()

df = pd.read_csv(dataset)
df['line_no'] = df.index.values

x_cols = ['x_'+col for col in list(df)]
y_cols = list(df)
df[x_cols] = None

meta_data_cols = [col for col in list(df) if (text_col not in col)]

GRAPH = ceda_model(
    sigma=1.,
    device='cuda',
    wv_model='roberta-base',
    wv_layers=level
)

for k,i in enumerate(df.index):

    if k >= start_at:
        df[x_cols] = df[y_cols].loc[i].to_list()
        output_name_ = output_name.format((k+1))

        ### Set selection criteria columns ###
        # sel = (( ) | ( ) | ( )) & ()
        # GRAPH.fit(df['x_'+text_col].loc[sel].values.tolist(), df[text_col].loc[sel].values.tolist())
        # GRAPH.meta_data += df[meta_data_cols].loc[sel].to_dict(orient='records')

        ### No selection criterion needed ###
        GRAPH.fit(df['x_'+text_col].values.tolist(), df[text_col].values.tolist())
        GRAPH.meta_data += df[meta_data_cols].to_dict(orient='records')

        pd.DataFrame([{'line_no': k}]).to_csv(TRACKER_NAME, index=False, header=False, encoding='utf-8', mode='a')

    if (((k+1) % save_at) == 0) and (not os.path.exists(output_name_)):
        GRAPH.checkpoint(output_name_)

        GRAPH = ceda_model(
            sigma=1.,
            device='cuda',
            wv_model='roberta-base',
            wv_layers=level
        )

print('=======][=======\n')
