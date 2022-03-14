
import streamlit as st
import argparse
import tqdm
import numpy as np
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, BertForMaskedLM, AutoModel
from google.cloud import storage
from smart_open import open as sopen # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import copy
from sklearn.linear_model import Ridge

@st.cache(allow_output_mutation=True)
def load_vecs():

    vecs = np.load("encoding-wikipedia_10k_bert-base-uncased_mean.npy")
    with open("wiki.1m.raw.txt", "r") as f:
       lines = f.readlines()
    sents = [l.strip() for l in lines]

    return vecs, sents

    
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(model_name):
    print("loading model.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("done loading model.")
    return tokenizer, model

def encode(sentence, model, tokenizer, pooling):
    #st.write(sentence)
    #st.write(tokenizer.tokenize(sentence))
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence, add_special_tokens=True))
    #st.write(tokens)
    tokens_tensor = torch.Tensor([tokens]).long()
    with torch.no_grad():
        last_hidden_states = model(tokens_tensor)["last_hidden_state"][0]
    
    if pooling == "cls":
        h = last_hidden_states[0,:] 
    else:
        h = last_hidden_states.mean(dim=0)
    
    return h.detach().cpu().numpy()

def get_closet_neighbors_idx(vecs, h, n=1000):

	sims = cosine_similarity([h], vecs)[0]
	idx = (-sims).argsort()
	return idx[:n]

# load vectors to cache
wiki_vecs, wiki_sents = load_vecs()
#wiki_vecs = wiki_vecs / np.linalg.norm(wiki_vecs, axis=1, keepdims=True)
lengths = np.array([len(s.split(" ")) for s in wiki_sents])
# start demo    
    
model_str = "bert"
model_name = "bert-base-uncased" #"sentence-transformers/all-mpnet-base-v2"
pooling = "mean"


tokenizer,model = load_model_and_tokenizer(model_name)
_, similarity_model = load_model_and_tokenizer(model_name)

input_sent = st.text_input("Enter a sentence.", "COVID enters the cell by attaching itself to a cell-protein.")
to_remove = st.selectbox("Which information to remove?", ["length", "length-classification", "depth", "position", "relative-position"])
if to_remove == "length":
	P = np.load("P_length_{}.npy".format(model_str))
	ridge = Ridge()
	ridge.fit(wiki_vecs, lengths)
	print("length score:", ridge.score(wiki_vecs, lengths))
	ridge.fit(wiki_vecs@P, lengths)
	print("length score projected:", ridge.score(wiki_vecs@P, lengths))
elif to_remove == "length-classification":
	P = np.load("P_length_clf_{}.npy".format(model_str))
	ridge = Ridge()
	ridge.fit(wiki_vecs, lengths)
	print("length score:", ridge.score(wiki_vecs, lengths))
	ridge.fit(wiki_vecs@P, lengths)
	print("length score projected:", ridge.score(wiki_vecs@P, lengths))
elif to_remove == "depth":
	P = np.load("P_depth_{}.npy".format(model_str))
elif to_remove == "position":
	P = np.load("P_position_{}.npy".format(model_str))
elif to_remove == "relative-position":
	P = np.load("P_relative_position_{}.npy".format(model_str))


#h = encode(input_sent, model, tokenizer,pooling)
tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_sent, add_special_tokens=True))
tokens_tensor = torch.Tensor([tokens]).long()

with torch.no_grad():
    last_hidden_state = similarity_model(tokens_tensor)["last_hidden_state"][0].mean(dim=0).detach().cpu().numpy()


n=8
idx = get_closet_neighbors_idx(wiki_vecs@P, P@last_hidden_state, n=n)
idx_original = get_closet_neighbors_idx(wiki_vecs, last_hidden_state, n=n)
sents = [wiki_sents[i] for i in idx]
sents_original = [wiki_sents[i] for i in idx_original]

st.write("Original:")
st.table(sents_original)
st.write("After:")
st.table(sents)
