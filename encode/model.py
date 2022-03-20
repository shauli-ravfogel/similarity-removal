import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, AutoTokenizer, AutoModel
import numpy as np
from typing import List
import tqdm

class Encoder(object):
    
    def __init__(self, device = 'cpu', model_name = "roberta-base"):
        
        #self.tokenizer = BertTokenizer.from_pretrained('scibert_scivocab_uncased/vocab.txt')
        #self.model = BertModel.from_pretrained('scibert_scivocab_uncased/')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
            
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        
    def tokenize_and_pad(self, texts: List[str]):

        tokenizer_out = self.tokenizer.batch_encode_plus(texts, add_special_tokens=True, padding=True, truncation=True,max_length=512, return_tensors="pt")
        input_ids, att_mask = tokenizer_out["input_ids"], tokenizer_out["attention_mask"]
        return input_ids.to(self.device), att_mask.to(self.device)        
        
    
    def encode(self, sentences: List[str], sentence_ids: List[str], batch_size: int, strategy: str = "mean", fname="", write = False):
        assert len(sentences) == len(sentence_ids)
        vecs = []
        

        for batch_idx in tqdm.tqdm(range(0, len(sentences), batch_size), total = len(sentences)//batch_size, ascii=True):
            
                batch_sents = sentences[batch_idx: batch_idx + batch_size]
                batch_ids = sentence_ids[batch_idx: batch_idx + batch_size]
                assert len(batch_sents) == len(batch_ids)
                
                idx, att_mask = self.tokenize_and_pad(batch_sents)
            
                with torch.no_grad():
                    outputs = self.model(idx, attention_mask = att_mask)
                    last_hidden = outputs[0]
                
                    if strategy == "cls":
                        h = last_hidden[:, 0, ...]
                    elif strategy == "mean":
                        last_hidden = last_hidden * att_mask[:,:,None] # zero out the padded states
                        lengths = att_mask.sum(dim=1, keepdims=True)
                        h = torch.sum(last_hidden, dim = 1) / lengths
                    elif strategy == "none":
                        h = last_hidden            
                batch_np = h.detach().cpu().numpy()
                assert len(batch_np) == len(batch_sents)
                
                sents_states_ids = zip(batch_sents, batch_np, batch_ids)
                for sent, vec, sent_id in sents_states_ids:
                    
                    #vec_str = " ".join(["%.4f" % x for x in vec])
                    #sent_dict = {"text": sent, "vec": vec_str, "id": sent_id}
                    vecs.append(vec)
        return vecs
