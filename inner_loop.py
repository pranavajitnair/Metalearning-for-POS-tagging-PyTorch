import torch
import pyconll
from models import POSTagger,Word
from functions import preprocess,preprocess_2
from collections import OrderedDict


class InnerLoop:
        def __init__(self,lossFunction,epochs,file_location,hidden_size,lang,word_number):
                
                file_location=file_location
                sentences=pyconll.load_from_file(file_location)
                sentences=preprocess_2(sentences)
                
                self.words=Word(sentences)
                self.words.addWords()
               
                self.x_train,self.y_train=preprocess(sentences,self.words)
                self.lossFunction=lossFunction
                self.encoder=POSTagger(word_number,hidden_size,self.words.n_tokens,116)
                self.epochs=epochs
                            
        def train(self,weights,batch_number):
                print("HA")
                
                for j in range(self.epochs):
                        output,_=self.encoder(torch.tensor(self.x_train[j+batch_number]),weights)
                        loss=self.lossFunction(output,torch.tensor(self.y_train[j+batch_number])) 
                        grads=torch.autograd.grad(loss,weights.values(),create_graph=True,allow_unused=True)
                        weights=OrderedDict((name,param-self.func(grad,param)) for ((name,param),grad) in zip(weights.items(), grads))
                        
                meta_grads = {name:g for ((name, _),g) in zip(weights.items(),grads)}
                return meta_grads,loss/(len(self.x_train))
            
        def func(self,grad,param):
                if grad is not None:
                        return grad
                else:
                        return param/5.0