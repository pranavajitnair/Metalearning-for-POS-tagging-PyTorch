import torch
import keras
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os
import pyconll

file_location='/home/pranav/Pictures/Hindi/hi_hdtb-ud-train.conllu'
sentences = pyconll.load_from_file(file_location)

class Word:
        def __init__(self,data):
                self.n_words=0
                self.n_tokens=16
                self.word_to_int={}
                self.int_to_word={}
                self.data=data
                self.token_to_int={}
                self.int_to_token={}
        
        def addWords(self):
                s=set()
                s.add('PAD')
                k=set()
                for sentence in self.data:
                        for token in sentence:
                                s.add(token.form)
                                k.add(token.upos)
                s=list(s)
                k=list(k)
                self.n_words=len(s)
                self.create_dict(s,k)
                
        def create_dict(self,s,k):
                for i in range(len(s)):
                        self.int_to_word[i]=s[i]
                        self.word_to_int[s[i]]=i
                        
                for i in range(len(k)):
                        self.int_to_token[i]=k[i]
                        self.token_to_int[k[i]]=i
                        
words=Word(sentences)
words.addWords()

def preprocess(data):
        xtrain=[]
        ytrain=[]
        for sentence in data:
                a=[]
                b=[]
                for token in sentence:
                        a.append(words.word_to_int[token.form])
                        b.append(words.token_to_int[token.upos])
                for _ in range(116-len(a)):
                        a.append(words.word_to_int['PAD'])
                        b.append(words.token_to_int['X'])
                xtrain.append(a)
                ytrain.append(b)
        return xtrain,ytrain
        

class POSTagger(nn.Module):
    
        def __init__(self,n_words,h_size,n_tokens,max_len):
                super(POSTagger,self).__init__()
                self.Dense=nn.Linear(h_size,n_tokens)
                self.lstm=nn.GRU(h_size,h_size)
                self.embedding=nn.Embedding(n_words,h_size)
                self.max_len=max_len
                
        def forward(self,input,hidden=None):
                input=self.embedding(input).view(1,self.max_len,-1)
                output,hidden=self.lstm(input,hidden)
                output=self.Dense(output)
                output=output.squeeze()
                return output,hidden

x_train,y_train=preprocess(sentences)

def train(x_train,y_train,encoder,optimizer,lossFunction):
          optimizer.zero_grad()
          loss=0
          for i in range(len(x_train)):
                  output,hidden=encoder(torch.tensor(x_train[i]))
                  loss+=lossFunction(output,torch.tensor(y_train[i]))
          loss.backward()
          optimizer.step()
          
          return loss.item()/len(x_train)
      
def train_iters(encoder,epochs,lossFunction,optimizer,datax,datay):
       
        for i in range(epochs):
                loss=train(datax,datay,encoder,optimizer,lossFunction)
                print('%f %d' %(loss,i))

epochs=20
max_len=116      
hidden_size=50
encoder=POSTagger(words.n_words,hidden_size,words.n_tokens,max_len)
lossFunction=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(encoder.parameters(),lr=0.01)                
train_iters(encoder,epochs,lossFunction,optimizer,x_train,y_train)                    