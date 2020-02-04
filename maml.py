from inner_loop import InnerLoop
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import POSTagger
from collections import OrderedDict
import gensim.models as gs
from data_loader import DataLoader,get_tokens,get_sentences,load_sentences

                        
class MetaLearn:
        def __init__(self,hindi_data_loader,marathi_data_loader,lossFunction,hidden_size,epochs,inner_epoch,max_len,n_tokens,tokens_dict,dict_token):
                
                self.hindi=InnerLoop(lossFunction,inner_epoch,hidden_size,n_tokens,hindi_data_loader)
                self.marathi=InnerLoop(lossFunction,inner_epoch,hidden_size,n_tokens,marathi_data_loader)
                self.hidden_size=hidden_size
                self.epochs=epochs
                self.encoder=POSTagger(hidden_size,n_tokens)
                self.optimizer=optim.Adam(self.encoder.parameters(),lr=0.01)
                self.lossFunction=lossFunction
                self.max_len=max_len
                self.inner_epoch=inner_epoch
                self.n_tokens=n_tokens
                self.token_to_index=tokens_dict
                self.index_to_token=dict_token
                
        def meta_update1(self,grads,print_epoch):
              
                x_val,y_val=self.marathi.data_loader.load_next()
               
                output,hidden=self.encoder(x_val)
                loss=self.lossFunction(output,y_val)
                
                hooks = []
                for (k,v) in self.encoder.named_parameters():
                        def get_closure():
                                key = k
                                def replace_grad(grad):
                                        return grads[key]
                                return replace_grad
                        hooks.append(v.register_hook(get_closure()))
                self.optimizer.zero_grad()        
                loss.backward()
                print(str(print_epoch)+" "+str(loss.item()))
                self.optimizer.step()

                for h in hooks:
                        h.remove()
                        
        def meta_update2(self,loss,print_epoch):
                self.optimizer.zero_grad()    
                loss.backward()
                print(str(print_epoch)+" "+str(loss.item()))
                self.optimizer.step()
                
        def train(self):
                for epoch in range(self.epochs):
                        fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                        
                        grad1,loss1=self.marathi.train(fast_weights)
                        grad2,loss2=self.hindi.train(fast_weights)
                        
                        grads=OrderedDict()
                        l=['lstm.weight_ih_l0','lstm.weight_hh_l0','lstm.bias_ih_l0','lstm.bias_hh_l0','Dense.weight','Dense.bias']
                        for k in l:
                                grads[k]=grad1[k]+grad2[k]
                                
                        self.meta_update1(grads,epoch)
                        #self.meta_update2(loss1+loss2,epoch)
                        
        def test(self,t):
                for _ in range(t,self.inner_epoch+t):
                        x_test,y_test,sentence=self.marathi.data_loader.load_next_test()
                        output,hidden=self.encoder(x_test)
                        loss=self.lossFunction(output,y_test)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        
                for _ in range(t):
                        x_test,y_test,sentence1=self.marathi.data_loader.load_next_test()
                        output,hidden=self.encoder(x_test)
                        output=F.softmax(output,dim=1)
                        
                        j=0
                        sentence=''
                        s=''
                        spredict=''

                        outputprime=[]
                        for k in range(self.max_len):
                                ma=0
                                cal=0
                                for j in range(self.n_tokens):
                                        if output[k][j]>ma:
                                            ma=output[k][j]
                                            cal=j
                                outputprime.append(cal)
                                
                        j=0
                        count=0
                        while sentence1[j]!='EOS':
                                s=s+self.index_to_token[int(y_test[j])]+' '
                                sentence=sentence+sentence1[j]+' '
                                spredict=spredict+self.index_to_token[outputprime[j]]+' '
                                if self.index_to_token[outputprime[j]]==self.index_to_token[int(y_test[j])]:
                                        count+=1
                                j+=1
                                
                        print(sentence)
                        print(s)
                        print(spredict)
                        print('Accuracy= '+str(100*count/j)+'%')
                        print('')
                        
                        
lossFunction=nn.CrossEntropyLoss()
hidden_size=1024
epochs=100
inner_epoch=15
test_size=10
max_len=116

marathi_train,marathi_test,hindi_train,hindi_test=load_sentences()
tokens_dict,dict_token,n_tokens=get_tokens(marathi_train)

marathi_train,marathi_test,marathi_train_tags,marathi_test_tags=get_sentences(marathi_train,marathi_test,tokens_dict,max_len)
hindi_train,hindi_test,hindi_train_tags,hindi_test_tags=get_sentences(hindi_train,hindi_test,tokens_dict,max_len)

model_hindi=gs.Word2Vec(hindi_train+hindi_test,min_count=1,size=hidden_size)
model_marathi=gs.Word2Vec(marathi_test+marathi_train,min_count=1,size=hidden_size)

hindi_data_loader=DataLoader(hindi_train,hindi_test,hindi_train_tags,hindi_test_tags,max_len,model_hindi)
marathi_data_loader=DataLoader(marathi_train,marathi_test,marathi_train_tags,marathi_test_tags,max_len,model_marathi)


metaLearn=MetaLearn(hindi_data_loader,marathi_data_loader,lossFunction,hidden_size,epochs,inner_epoch,max_len,n_tokens,tokens_dict,dict_token)
metaLearn.train()
metaLearn.test(test_size)   