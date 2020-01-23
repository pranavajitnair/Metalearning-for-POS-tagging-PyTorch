import torch
import pyconll
from models import POSTagger,Word
from functions import preprocess,preprocess_2
from collections import OrderedDict
import torch.optim as optim


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
                self.optimizer=optim.SGD(self.encoder.parameters(),lr=0.01)
                            
        def train(self,weights,batch_number):
                self.encoder.load_state_dict(weights)
                for j in range(self.epochs):
                        self.optimizer.zero_grad()
                        output,_=self.encoder(torch.tensor(self.x_train[j+batch_number]))
                        loss=self.lossFunction(output,torch.tensor(self.y_train[j+batch_number]))
                        loss.backward()
                        self.optimizer.step()
                        
                meta_grads =OrderedDict((name,param.grad) for (name,param) in self.encoder.named_parameters())
                return meta_grads,loss
            