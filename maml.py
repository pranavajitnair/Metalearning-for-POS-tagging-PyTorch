import torch
import torch.optim as optim

from collections import OrderedDict
import os

from inner_loop import CRF_BiLSTM

                        
class MetaLearn(object):
        def __init__(self,hindi_data_loader,marathi_data_loader,lossFunction,hidden_size,epochs,inner_epoch,max_len,n_tokens,tokens_dict,dict_token,char_dict,n_chars,learning_rate):
                
                self.hindi=CRF_BiLSTM(inner_epoch,hidden_size,n_tokens,hindi_data_loader,tokens_dict,char_dict,n_chars) #.cuda()
                self.marathi=CRF_BiLSTM(inner_epoch,hidden_size,n_tokens,marathi_data_loader,tokens_dict,char_dict,n_chars) #.cuda()
                self.hidden_size=hidden_size
                
                self.epochs=epochs
                self.encoder=CRF_BiLSTM(inner_epoch,hidden_size,n_tokens,marathi_data_loader,tokens_dict,char_dict,n_chars) #.cuda()
                self.optimizer=optim.Adam(self.encoder.parameters(),lr=learning_rate)
                self.lossFunction=lossFunction
                
                self.max_len=max_len
                self.inner_epoch=inner_epoch
                self.n_tokens=n_tokens
                self.token_to_index=tokens_dict
                self.index_to_token=dict_token
                self.char_dict=char_dict
                self.n_chars=n_chars
                self.learning_rate=learning_rate
                
        def meta_update1(self,grads,print_epoch):
              
                x_val,y_val,sentence_text=self.marathi.data_loader.load_next()
                x,y,sentence=self.hindi.data_loader.load_next()

                loss1=self.encoder.test_train(sentence,x,y)
                loss=self.encoder.test_train(sentence_text,x_val,y_val)
                loss=loss+loss1
                
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
                print('epoch=',print_epoch+1,"training loss=",str(loss.item()))
                self.optimizer.step()

                for h in hooks:
                        h.remove()
                        
        def meta_update2(self,loss,print_epoch):
                self.optimizer.zero_grad()    
                loss.backward()
                print(str(print_epoch)+" "+str(loss.item()))
                self.optimizer.step()
                
        def train_MAML(self):
                for epoch in range(self.epochs):
                        fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                        ls=[]

                        grad1,loss1=self.marathi.train(fast_weights)
                        grad2,loss2=self.hindi.train(fast_weights)

                        ls.append(grad1)
                        ls.append(grad2)

                        grads={k: sum(d[k] for d in ls) for k in ls[0].keys()}
                        
                        self.meta_update1(grads,epoch)
                        #self.meta_update2(loss1+loss2,epoch)
                        
        def train_Reptile(self,epsilon):
                for epoch in range(self.epochs):
                        fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                        
                        weights_hindi,loss1=self.hindi.train(fast_weights,True,False)
                        weights_marathi,loss2=self.marathi.train(fast_weights,True,False)

                        for name,param in self.encoder.named_parameters():
                                param=param+epsilon*(weights_hindi[name]+weights_marathi[name]-2*param)

                        print('epoch=',epoch+1,'training loss=',loss1+loss2)
                        
        def train_FOMAML(self):
                for epoch in range(self.epochs):
                        fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                        
                        weights_hindi,loss1=self.hindi.train(fast_weights,False,True)
                        weights_marathi,loss2=self.marathi.train(fast_weights,False,True)

                        for name,param in self.encoder.named_parameters():
                                param=param-0.1*(weights_hindi[name]+weights_marathi[name])

                        print('epoch=',epoch+1,'training loss=',loss1+loss2)
                        
        def test(self,t,lang):
                for _ in range(self.inner_epoch):
                        if lang=='marathi':
                                x_test,y_test,sentence=self.marathi.data_loader.load_next()
                        else:
                                x_test,y_test,sentence=self.hindi.data_loader.load_next()
                                
                        loss=self.encoder.test_train(sentence,x_test,y_test)
                        loss.backward()
                        self.optimizer.step()
                
                a=0
                b=0
                c=0
                for _ in range(t):
                        if lang=='marathi': 
                                x_test,y_test,sentence1=self.marathi.data_loader.load_next_test()
                        else:        
                                x_test,y_test,sentence1=self.hindi.data_loader.load_next_test()
                                
                        score,outputprime=self.encoder.forward(x_test,sentence1)
                        
                        j=0
                        count=0
                        for i in range(len(sentence1)):
                                c+=1                            
                                if outputprime[j]==y_test[j]:
                                        count+=1
                                        b+=1
                                j+=1
                        
                        accuracy=100*count/j
                        a+=accuracy
                        
                print(a/t)
                print(b*100/c)
                
        def store_checkpoint(self,finished_epochs,epsilon=None):
                dict={}
                dict['epochs']=self.epochs-finished_epochs
                dict['model']=self.encoder.state_dict()
                dict['dataloader_hindi']=self.hindi.data_loader
                dict['dataloader_marathi']=self.marathi.data_loader
                dict['n_tokens']=self.n_tokens
                dict['token_to_index']=self.token_to_index
                dict['index_to_token']=self.index_to_token
                dict['max_len']=self.max_len
                dict['hidden_size']=self.hidden_size
                dict['char_dict']=self.char_dict
                dict['learning_rate']=self.learning_rate
                dict['n_chars']=self.n_chars
                dict['inner_epoch']=self.inner_epoch
                dict['epsilon']=epsilon
                torch.save(dict,os.getcwd()+'/checkpoints/model.pth')