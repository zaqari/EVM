import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

class languageModel(nn.Module):

    def __int__(self, vector_model: object, tokenizer: object):
        super(languageModel, self).__int__()
        self.wv = vector_model
        self.tok = tokenizer

    def forward(self, text):
        inputs = self.tok(text, return_tensors='pt')
        return self.wv(**inputs)