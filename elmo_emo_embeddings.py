# Imports
import allennlp
import os
import csv
import json
import nltk
import torch
from collections import Counter
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from collections import defaultdict
import math
from utils_emo import *
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from os import path

# Read in Data
print("starting elmo emo embedding script")

if(path.exists("responses.npy") and path.exists("train_labels.npy")):
    responses = np.load('responses.npy')
    responses = responses[0:60000]
    train_labels = np.load('train_labels.npy')
    train_labels = train_labels[0:60000]
    print("responses and labels already exist")
    print("..loading emo separator")
else:
    print("..loading responses")
    responses, train_labels = get_textual_examples('../SARC/2.0/main')
    responses = responses[0:60000]
    train_labels = train_labels[0:60000]
    
print(len(responses))
print("done formatting data")

emo_responses, emo_vectors = sep_emo(responses)
responses = emo_responses

com_emo = [cm[0] for cm in emo_vectors]
rep_emo = [rp[1] for rp in emo_vectors]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize ElmoEmbedder
elmo = ElmoEmbedder()

# Run batches of sentences through the ElmoEmbedder
print("starting to run batches of examples through the ElmoEmbedder")
batch_size = 64
num_batches = math.ceil(len(responses)/batch_size)
all_example_embeddings = []
print('num batches', num_batches)
for i in range(num_batches):
    batch_examples = []
    if i % 100 == 0:
        print('on batch', i)
    if i == num_batches - 1:
        batch_examples = responses[batch_size*i:]
        com_emo_batch = com_emo[batch_size*i:]
        rep_emo_batch = rep_emo[batch_size*i:]

    else:
        batch_examples = responses[batch_size*i:batch_size*(i+1)]
        com_emo_batch = com_emo[batch_size*i:batch_size*(i+1)]
        rep_emo_batch = rep_emo[batch_size*i:batch_size*(i+1)]
    
    first_sentence_embeddings = []
    second_sentence_embeddings = []
    
    first_sentences = [nltk.word_tokenize(x[0]) for x in batch_examples]
    second_sentences = [nltk.word_tokenize(x[1]) for x in batch_examples]
    
    first_batch = elmo.embed_batch(first_sentences)
    second_batch = elmo.embed_batch(second_sentences)
    
    #print(type(first_batch))
    #print(np.array(first_batch).shape)
    #print(first_batch[0])
    #print(len(first_batch[0])) 3
    #print(len(first_batch[0][0])) 5
    #print(len(first_batch[0][0][0]))
    
    
    for j, sent in enumerate(first_batch):
        #print('shape:', sent.shape)
        #break
        first_sent_embedding = np.mean(sent[2], axis = 0)
        first_sentence_embeddings.append(list(first_sent_embedding))
        second_sent_embedding = np.mean(second_batch[j][2], axis = 0)
        second_sentence_embeddings.append(list(second_sent_embedding))
    example_embeddings = [first_sentence_embeddings[k] + com_emo_batch[k] + second_sentence_embeddings[k] + rep_emo_batch[k] for k in range(len(first_sentence_embeddings))]
    all_example_embeddings += example_embeddings


print("starting to save elmo-emo-X file")
np.save('balanced-elmo-emo-X.npy', all_example_embeddings)

print("saved elmo-emo-X file")
labels = [int(x[0]) for x in train_labels]

print("starting to save elmo-emo-Y file")
np.save('balanced-elmo-emo-Y.npy', labels)

print("saved elmo-emo-Y file")