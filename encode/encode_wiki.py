from model import Encoder
import pickle
import numpy as np
import torch


with open("wiki.1million.raw.txt", "r") as f:
    data = f.readlines()

#data = data[:1000]

#model_name="roberta-base"
model_name = "bert-base-uncased"
batch_size=128
strategy="mean"

model = Encoder(model_name=model_name,device="cuda:0")
vecs = model.encode(data[:], sentence_ids=list(range(len(data[:]))), batch_size=batch_size, strategy=strategy)

#assert len(data) == len(vecs)

with open('encoding/wikipedia_1m_{}_{}.npy'.format(model_name, strategy).replace("/","-"), 'wb') as f:

    np.save(f, vecs)

