import torch
import torch.nn as nn
import numpy as np
import nltk.data
import re

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def sentences(text):
    sents = sent_detector.tokenize(text)
    sents = [re.sub(r'[^\w\s]',' ', sent) for sent in sents]
    return sents

class analyzer(nn.Module):

    def __init__(self, wv: object, sim_model: object, device: str='cuda', x_kw:str='', y_kw: str=''):
        super(analyzer, self).__init__()
        self.dev = device
        self.wv = wv
        self.sim = sim_model
        self.x_text = None
        self.ex = None
        self.y_text = None
        self.ey = None

        # keywords to look for
        self.xkw = x_kw
        self.ykw = y_kw

    def __process_text(self, x, look_for_kw: str=''):
        texts = sentences(x)
        VECS = []
        for text in texts:
            try:

                if look_for_kw:
                  text = str(text).replace(look_for_kw, ' [XxX] ' + look_for_kw + ' [XxX]')

                w, tokens = self.wv._tokenize(str(text).replace('\n', ' '))
                sel = (
                        (
                                tokens == np.array(
                            ['.', '!', '?', ',', '...', ':', ';', '. . .', '>', 'Ġ', ' ', '#']
                        ).reshape(-1,1)
                        ).sum(axis=0) == 0
                )

                # sets input id & token list dimensions
                w = np.array(w)[sel]
                tokens = tokens[sel]

                # defines the range for tokens related to the kw search
                if look_for_kw:
                    locs = (tokens == '[XxX]').nonzero()[0].reshape(-1, 2)
                    locs -= np.array([(2 * i) for i in range(len(locs))]).reshape(-1, 1)
                    locs[:, -1] -= 1

                # creates vectors
                vecs = self.wv.E(torch.LongTensor(w[tokens!='[XxX]']).view(-1))

                if look_for_kw:
                    VECS += [torch.cat([vecs[s:e] for s,e in locs], dim=0)]
                else:
                    VECS += [vecs]

            except Exception:
                None

        if len(VECS) > 0:
            return torch.cat(VECS, dim=0)
        else:
            return torch.zeros(size=(1, self.wv.mod.embeddings.word_embeddings.weight.shape[-1]*len(self.wv.layers))).float()

    def forward(self, x=None, y=None):

        #The code below may seem strange, but it's quite helpful.
        # In cases where analyzer is used recursively, this snippet
        # of code specifically only creates a new self.ex entry iff
        # it is the case that the text provided in x is different
        # from the prior text given for x.
        if (x != None) and (x != self.x_text):
            self.x_text = x
            self.ex = self.__process_text(x, self.xkw).to(self.dev)

        #And while it is NOT necessary, we repeat the same process
        # for y, just in case.
        if (y != None) and (y != self.y_text):
            self.y_text = y
            self.ey = self.__process_text(y, self.ykw).to(self.dev)

        if self.x_text and self.y_text:
            return self.sim(self.ex, self.ey), (self.ex.shape[0], self.ey.shape[0])

        else:
            return torch.FloatTensor([-404.404, -404.404]), (0, 0)

