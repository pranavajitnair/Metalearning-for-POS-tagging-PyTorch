from inner_loop import InnerLoop
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import POSTagger
import torch
from collections import OrderedDict
import os

                        
class MetaLearn:
        def __init__(self,file_location_for_hindi,file_location_for_marathi,lossFunction,hidden_size,epochs,inner_epoch,max_len):
                
                self.hindi=InnerLoop(lossFunction,inner_epoch,file_location_for_hindi,hidden_size,'hindi',None,max_len)
                self.marathi=InnerLoop(lossFunction,inner_epoch,file_location_for_marathi,hidden_size,'marathi',self.hindi.words.n_words,max_len)
                self.hidden_size=hidden_size
                self.epochs=epochs
                self.encoder=POSTagger(self.hindi.words.n_words,hidden_size,self.hindi.words.n_tokens,max_len)
                self.optimizer=optim.SGD(self.encoder.parameters(),lr=0.01)
                self.lossFunction=lossFunction
                self.max_len=max_len
                self.inner_epoch=inner_epoch
                
        def meta_update(self,grads,i,print_epoch,loss):
              
#                x_val=self.marathi.x_train[i]
#                y_val=self.marathi.y_train[i]
#                
#                output,hidden=self.encoder(torch.tensor(x_val))
#                loss=self.lossFunction(output,torch.tensor(y_val))
#                
#                hooks = []
#                for (k,v) in self.encoder.named_parameters():
#                        def get_closure():
#                                key = k
#                                def replace_grad(grad):
#                                        return grads[key]
#                                return replace_grad
#                        hooks.append(v.register_hook(get_closure()))
                self.optimizer.zero_grad()        
                loss.backward()
                print(str(print_epoch)+" "+str(loss.item()))
                self.optimizer.step()

#                for h in hooks:
#                        h.remove()
                
        def train(self):
                for epoch in range(self.epochs):
                        fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                        
                        grad1,loss1=self.marathi.train(fast_weights,(epoch*self.inner_epoch)%200)
                        grad2,loss2=self.hindi.train(fast_weights,epoch*self.inner_epoch)
                        
                        grads=OrderedDict()
                        l=['embedding.weight','lstm.weight_ih_l0','lstm.weight_hh_l0','lstm.bias_ih_l0','lstm.bias_hh_l0','Dense.weight','Dense.bias']
                        for k in l:
                                grads[k]=grad1[k]+grad2[k]
                                
                        self.meta_update(grads,(epoch*self.inner_epoch)%200,epoch,loss1+loss2)
                        
        def test(self,t):
                for i in range(t,self.inner_epoch+t):
                        output,hidden=self.encoder(torch.tensor(self.marathi.x_train[i]))
                        loss=self.lossFunction(output,torch.tensor(self.marathi.y_train[i]))
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        
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
hidden_size=1024
epochs=20
inner_epoch=15
test_size=10
max_len=116

metaLearn=MetaLearn(file_location_for_hindi,file_location_for_marathi,lossFunction,hidden_size,epochs,inner_epoch,max_len)
metaLearn.train()
metaLearn.test(test_size)
