import torch
from models import POSTagger
from collections import OrderedDict
import torch.optim as optim


class InnerLoop:
       def __init__(self,lossFunction,epochs,hidden_size,n_tokens,data_loader):
            
               self.lossFunction=lossFunction
               self.encoder=POSTagger(hidden_size,n_tokens)
               self.epochs=epochs
               self.optimizer=optim.SGD(self.encoder.parameters(),lr=0.01)
               self.data_loader=data_loader
                           
       def train(self,weights):
               for j in range(self.epochs):
                       self.encoder.load_state_dict(weights)
                       sentence,tags=self.data_loader.load_next()
                       output,_=self.encoder(sentence)
                       loss=self.lossFunction(output,tags)
                       grads=torch.autograd.grad(loss,self.encoder.parameters(),create_graph=True)
                       weights=OrderedDict((name, param - 0.01*grad) for ((name, param), grad) in zip(weights.items(), grads))
                       
               self.encoder.load_state_dict(weights)
               sentence,tags=self.data_loader.load_next()
               output,_=self.encoder(sentence)
               loss=self.lossFunction(output,tags)
               grads=torch.autograd.grad(loss,self.encoder.parameters(),create_graph=True)
               meta_grads={name:g for ((name, _), g) in zip(self.encoder.named_parameters(), grads)}
                       
               return meta_grads,loss