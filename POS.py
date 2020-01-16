import numpy as np
import random
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import treebank
import torch

data=treebank.tagged_sents(tagset='universal')

s=set()
s1=set()

for sentences in data:
        sentence=''
        for words in sentences:
                sentence+=words[0]+' '
                s1.add(words[0])
                s.add(words[1])
                
s1=list(s1)
s=list(s)
s1.append('EOS')

dict={}
dict1={}

for i in range(len(s)):
        dict[s[i]]=i
        
for i in range(len(s1)):
        dict1[s1[i]]=i
        
datay=[]
datax=[]

for sentences in data:
        sentence=[]
        tags=[]
        if len(sentences)>25:
                continue
        for words in sentences:
                sentence.append(dict1[words[0]])
                tags.append(dict[words[1]])
        for _ in range(len(tags),25):
                sentence.append(dict1['EOS'])
                tags.append(dict['X'])
        datax.append(sentence)
        datay.append(tags)
        
        
class POSTagger(nn.Module):
    
        def __init__(self):
                super(POSTagger,self).__init__()
                self.Dense=nn.Linear(50,12)
                self.lstm=nn.GRU(50,50)
                self.embedding=nn.Embedding(len(s1),50)
                
        def forward(self,input,hidden=None):
                input=self.embedding(input).view(1,25,-1)
                output,hidden=self.lstm(input,hidden)
                output=self.Dense(output)
                output=output.squeeze()
                return output,hidden

def train(x_train,y_train,encoder,optimizer,lossFunction):
          optimizer.zero_grad()
          loss=0
          for i in range(len(x_train)):
                  output,hidden=encoder(torch.tensor(x_train[i]))
                  loss+=lossFunction(output,torch.tensor(y_train[i]))
          loss.backward()
          optimizer.step()
          
          return loss.item()/len(x_train)
         
def train_iters(encoder,epochs,lossFunction,optimizer):
       
        for i in range(epochs):
                loss=train(datax,datay,encoder,optimizer,lossFunction)
                print('%f %d' %(loss,i))

encoder=POSTagger()
epochs=100
lossFunction=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(encoder.parameters(),lr=0.01)                
train_iters(encoder,epochs,lossFunction,optimizer)

word=6
s2=''
i=0

output,_=encoder(torch.tensor(datax[word]))
output=F.softmax(output,dim=1)

while s1[datax[word][i]]!='EOS':
     s2+=s1[datax[word][i]]+' '
     i+=1       
print(s2)

outputprime=[]
for i in range(25):
        ma=0
        cal=0
        for j in range(12):
                if output[i][j]>ma:
                    ma=output[i][j]
                    cal=j
        outputprime.append(cal)
        
for i in range(len(outputprime)):
    print(s[outputprime[i]])
    
print("done")

for i in range(len(datay[word])):
        print(s[datay[word][i]])
        