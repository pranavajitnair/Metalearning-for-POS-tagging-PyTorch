import torch
import pyconll
from models import POSTagger,Word
from functions import preprocess,preprocess_2
from collections import OrderedDict
import torch.optim as optim


class InnerLoop:
       def __init__(self,lossFunction,epochs,file_location,hidden_size,lang,word_number,max_len):
            
               file_location=file_location
               sentences=pyconll.load_from_file(file_location)
               sentences=preprocess_2(sentences)
               
               self.words=Word(sentences)
               self.words.addWords()
               if word_number is None:
                       word_number=self.words.n_words
               self.x_train,self.y_train=preprocess(sentences,self.words,max_len)
               self.lossFunction=lossFunction
               self.encoder=POSTagger(word_number,hidden_size,self.words.n_tokens,max_len)
               self.epochs=epochs
               self.optimizer=optim.SGD(self.encoder.parameters(),lr=0.01)
                           
       def train(self,weights,batch_number):
               for j in range(self.epochs):
                       self.encoder.load_state_dict(weights)
                       output,_=self.encoder(torch.tensor(self.x_train[j+batch_number]))
                       loss=self.lossFunction(output,torch.tensor(self.y_train[j+batch_number]))
                       grads=torch.autograd.grad(loss,self.encoder.parameters(),create_graph=True)
                       weights=OrderedDict((name, param - 0.01*grad) for ((name, param), grad) in zip(weights.items(), grads))
                       
               self.encoder.load_state_dict(weights)
               output,_=self.encoder(torch.tensor(self.x_train[j+batch_number]))
               loss=self.lossFunction(output,torch.tensor(self.y_train[j+batch_number]))
               grads=torch.autograd.grad(loss,self.encoder.parameters(),create_graph=True)
               meta_grads={name:g for ((name, _), g) in zip(self.encoder.named_parameters(), grads)}
                       
               return meta_grads,loss