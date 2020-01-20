from collections import OrderedDict
from inner_loop import InnerLoop
import torch.nn as nn

                        
class MetaLearn:
        def __init__(self,file_location_for_hindi,file_location_for_marathi,lossFunction,hidden_size,epochs):
                
                self.hindi=InnerLoop(lossFunction,epochs,file_location_for_hindi,hidden_size,'hindi')
                self.marathi=InnerLoop(lossFunction,epochs,file_location_for_marathi,hidden_size,'marathi',self.hindi.words.n_words)
                self.lossFunction=lossFunction
                self.hidden_size=hidden_size
                self.epochs=epochs
        
        def train(self):
                weights=OrderedDict((name,param) for (name,param) in self.hindi.encoder.named_parameters())
                
                for i in range(self.epochs):
                        grad1,loss1=self.hindi.train(weights)
                        grad2,loss2=self.marathi.train(weights)
                        
                        print(str(i)+" "+str(loss1+loss2))
                        
                        grads={}
                        for name,_ in grad1:
                                grads[name]=grad1[name]+grad2[name]
                        
                        weights=OrderedDict((name,param-0.01*grad) for ((name,param),grad) in zip(weights.items(), grads))
                

file_location_for_hindi='/home/pranav/Pictures/Hindi/hi_hdtb-ud-train.conllu'
file_location_for_marathi='/home/pranav/Pictures/Hindi/hi_hdtb-ud-train.conllu'
lossFunction=nn.CrossEntropyLoss()
hidden_size=10
epochs=10

metaLearn=MetaLearn(file_location_for_hindi,file_location_for_marathi,lossFunction,hidden_size,epochs)
metaLearn.train()