from inner_loop import InnerLoop
import torch.nn as nn
import torch.optim as optim
from models import POSTagger,Word
from functions import preprocess
import pyconll
import torch
from collections import OrderedDict

                        
class MetaLearn:
        def __init__(self,file_location_for_hindi,file_location_for_marathi,lossFunction,hidden_size,epochs):
                
                self.hindi_sentences=pyconll.load_from_file(file_location_for_hindi)
                self.words=Word(self.hindi_sentences)
                self.words.addWords()
                self.x_train,self.y_train=preprocess(self.hindi_sentences,self.words)
                self.marathi=InnerLoop(lossFunction,epochs,file_location_for_marathi,hidden_size,'marathi',self.words.n_words)
                self.lossFunction=lossFunction
                self.hidden_size=hidden_size
                self.epochs=epochs*10
                self.encoder=POSTagger(self.words.n_words,hidden_size,self.words.n_tokens,116)
                self.optimizer=optim.SGD(self.encoder.parameters(),lr=0.01)
                
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

file_location_for_hindi='/home/pranav/Pictures/Hindi/hi_hdtb-ud-train.conllu'
file_location_for_marathi='/home/pranav/Pictures/Hindi/hi_hdtb-ud-train.conllu'
lossFunction=nn.CrossEntropyLoss()
hidden_size=128
epochs=1000

metaLearn=MetaLearn(file_location_for_hindi,file_location_for_marathi,lossFunction,hidden_size,epochs)
metaLearn.train()