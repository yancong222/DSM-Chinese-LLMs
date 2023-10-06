import pandas as pd
import numpy as np
import re
import math
import os
import string
import ast
import scipy.stats
import shutil, sys, glob

# !pip install minicons
import torch
from minicons import cwe
# also allows gpus, use 'cuda:[NUMBER]' to do so.
BERT = cwe.CWE('bert-base-chinese', device = 'cuda') 
DeBERTa = cwe.CWE('IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese', device = 'cuda')
GPT2 = cwe.CWE('uer/gpt2-chinese-cluecorpussmall', device = 'cuda') 
XLNet = cwe.CWE('xlnet-base-cased', device = 'cuda') 

# define functions to extract embedding vectors
# 1 sentence as 1 context
# default last layer
def get_word_emb_from_sentences(word, sentences, model):
  word_embs = []
  instances = []
  sentences = ast.literal_eval(sentences)
  for sentence in sentences:
      instances.append([sentence, word])
      try:
        word_emb = model.extract_representation(instances)
        word_embs.append(np.array(word_emb[0]))
      except IndexError:
        #print('IndexError: ', instances)
        continue
      instances = []
      torch.cuda.empty_cache() #avoid 'CUDA out of memory' error
  # need to add axs = 0; otherwise it returns a float number like -0.8873 etc.
  return np.mean(word_embs, axis = 0) # get averaged sentence embeddings for the targe word

def get_word_emb_from_sentences_L4(word, sentences, model):
  last_four = list(range(model.layers+1))[-4:]
  word_embs = []
  instances = []
  sentences = ast.literal_eval(sentences)
  for sentence in sentences:
      instances.append([sentence, word])
      try:
        # get sum of emb from the last 4 layers
        word_emb = torch.stack(model.extract_representation(instances, layer = last_four)).sum(0) 
        word_embs.append(np.array(word_emb[0]))
      except IndexError:
        continue
      instances = []
      torch.cuda.empty_cache() 
  return np.mean(word_embs, axis = 0) 

def get_word_emb_from_sentences_F4(word, sentences, model):
  first_four = list(range(model.layers+1))[:4]
  word_embs = []
  instances = []
  sentences = ast.literal_eval(sentences)
  for sentence in sentences:
      instances.append([sentence, word])
      try:
        word_emb = torch.stack(model.extract_representation(instances, layer = first_four)).sum(0) 
        word_embs.append(np.array(word_emb[0]))
      except IndexError:
        continue
      instances = []
      torch.cuda.empty_cache() 
  return np.mean(word_embs, axis = 0) 

def get_word_emb_from_sentences_M4(word, sentences, model):
  word_embs = []
  instances = []
  sentences = ast.literal_eval(sentences)
  for sentence in sentences:
      instances.append([sentence, word])
      try:
        word_emb = torch.stack(model.extract_representation(instances, layer = [5,6,7,8])).mean(0)
        word_embs.append(np.array(word_emb[0]))
      except IndexError:
        continue
      instances = []
      torch.cuda.empty_cache() 
  return np.mean(word_embs, axis = 0) 

# extract embeddings
# using BERT as an example
# do the same for all the LLMs all the datasets
for i in df_emb.index:
    df_emb['word1_vec'] = get_word_emb_from_sentences(df_01_emb['word1'][i], df_01_emb['word1_sentences'][i], BERT)
    df_emb['word2_vec'] = get_word_emb_from_sentences(df_01_emb['word2'][i], df_01_emb['word2_sentences'][i], BERT)
    df_emb.to_csv(data + 'tbd.csv')
df_emb.tail()





