import torch
import torch.nn as nn
import torch.nn.functional as F
import pyconll
from models import POSTagger,Word
from functions import *
from collections import OrderedDict


class InnerLoop:
        def __init__(self,lossFunction,epochs,file_location,hidden_size,lang,word_number=None):
                
                file_location=file_location
                sentences=pyconll.load_from_file(file_location)
                sentences=preprocess_2(sentences)
                self.words=Word(sentences)
                self.words.addWords()
                
                if lang=='marathi':
                        word_count=word_number
                else:
                        word_count=self.words.n_words
                
               
                self.x_train,self.y_train=preprocess(sentences,self.words)
                self.lossFunction=lossFunction
                self.encoder=POSTagger(word_count,hidden_size,self.words.n_tokens,116)
                self.epochs=epochs
                
                            
        def train(self,weights):
                weights=weights
                for _ in range(self.epochs):
                        loss=0
                        for i in range(len(self.x_train)):
                                output,hidden=self.encoder(torch.tensor(self.x_train[i]),weights)
                                loss+=self.lossFunction(output,torch.tensor(self.y_train[i]))
                                
                        grads=torch.autograd.grad(loss,weights.values(),create_graph=True)
                        weights=OrderedDict((name,param-0.01*grad) for ((name,param),grad) in zip(weights.items(), grads))
                        
                meta_grads = {name:g for ((name, _),g) in zip(weights.items(),grads)}
                return meta_grads,loss/(len(self.x_train))