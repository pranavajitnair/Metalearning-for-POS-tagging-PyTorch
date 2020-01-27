from inner_loop import InnerLoop
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import POSTagger,Word
from functions import preprocess
import pyconll
import torch
from collections import OrderedDict
import os

                        
class MetaLearn:
        def __init__(self,file_location_for_hindi,file_location_for_marathi,lossFunction,hidden_size,epochs,inner_epoch,max_len):
                
                self.hindi_sentences=pyconll.load_from_file(file_location_for_hindi)
                self.words=Word(self.hindi_sentences)
                self.words.addWords()
                self.x_train,self.y_train=preprocess(self.hindi_sentences,self.words,max_len)
                self.marathi=InnerLoop(lossFunction,inner_epoch,file_location_for_marathi,hidden_size,'marathi',self.words.n_words,max_len)
                self.lossFunction=lossFunction
                self.hidden_size=hidden_size
                self.epochs=epochs
                self.encoder=POSTagger(self.words.n_words,hidden_size,self.words.n_tokens,max_len)
                self.optimizer=optim.SGD(self.encoder.parameters(),lr=0.01)
                self.max_len=max_len
                
        def meta_update(self,grads,i):
                x_val=self.x_train[i]
                y_val=self.y_train[i]
                
                output,hidden=self.encoder(torch.tensor(x_val))
                loss=self.lossFunction(output,torch.tensor(y_val))
                
                hooks = []
                for (k,v) in self.encoder.named_parameters():
                        def get_closure():
                                key = k
                                def replace_grad(grad):
                                        return grads[key]
                                return replace_grad
                        hooks.append(v.register_hook(get_closure()))
                        
                loss.backward()
                print(str(i)+" "+str(loss.item()))
                self.optimizer.zero_grad()
                self.optimizer.step()
                
                for h in hooks:
                         h.remove()
                
        def train(self):
                for epoch in range(self.epochs):
                        fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                        grads,loss=self.marathi.train(fast_weights,epoch)
                        self.meta_update(grads,epoch)
                        
        def test(self,t):
                for i in range(t):
                        output,hidden=self.encoder(torch.tensor(self.marathi.x_train[i]))
                        output=F.softmax(output,dim=1)
                        
                        j=0
                        sentence=''
                        s=''
                        spredict=''

                        outputprime=[]
                        for k in range(self.max_len):
                                ma=0
                                cal=0
                                for j in range(self.marathi.words.n_tokens):
                                        if output[k][j]>ma:
                                            ma=output[k][j]
                                            cal=j
                                outputprime.append(cal)
                                
                        j=0
                        count=0
                        while self.marathi.words.int_to_word[self.marathi.x_train[i][j]]!='PAD':
                                s=s+self.marathi.words.int_to_token[self.marathi.y_train[i][j]]+' '
                                sentence=sentence+self.marathi.words.int_to_word[self.marathi.x_train[i][j]]+' '
                                spredict=spredict+self.marathi.words.int_to_token[outputprime[j]]+' '
                                if self.marathi.words.int_to_token[outputprime[j]]==self.marathi.words.int_to_token[self.marathi.y_train[i][j]]:
                                        count+=1
                                j+=1
                                
                        print(sentence)
                        print(s)
                        print(spredict)
                        print('Accuracy= '+str(100*count/j)+'%')
                        print('')
                        
                        
file_location_for_hindi=os.getcwd()+'/hi_hdtb-ud-train.conllu'
file_location_for_marathi=os.getcwd()+'/mr_ufal-ud-train.conllu'
lossFunction=nn.CrossEntropyLoss()
hidden_size=512
epochs=1000
inner_epoch=15
test_size=10
max_len=116

metaLearn=MetaLearn(file_location_for_hindi,file_location_for_marathi,lossFunction,hidden_size,epochs,inner_epoch,max_len)
metaLearn.train()
metaLearn.test(test_size)