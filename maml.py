import torch
import torch.optim as optim
from torch.autograd import Variable

from collections import OrderedDict
import random

from inner_loop import CRF_BiLSTM
from data_loader import Data_Loader

                        
class MetaLearn(object):
        def __init__(self,hindi_data_loader,marathi_data_loader,sanskrit_data_loader,bhojpuri_data_loader
                     ,magahi_data_loader,english_data_loader,german_data_loader,dutch_data_loader,danish_data_loader
                     ,lossFunction,hidden_size,epochs,inner_epoch,max_len,n_tokens,tokens_dict,dict_token,char_dict,n_chars,N,K,lr):
                
                self.fast_net=CRF_BiLSTM(inner_epoch,hidden_size,n_tokens,tokens_dict,char_dict,n_chars) #.cuda()

                self.hidden_size=hidden_size
               
                self.encoder=CRF_BiLSTM(inner_epoch,hidden_size,n_tokens,tokens_dict,char_dict,n_chars) #.cuda()
                self.optimizer=optim.Adam(self.encoder.parameters(),lr=lr)
                self.lossFunction=lossFunction
                self.lr=lr
                
                self.max_len=max_len
                self.inner_epoch=inner_epoch
                self.epochs=epochs
                
                self.n_tokens=n_tokens
                self.token_to_index=tokens_dict
                self.index_to_token=dict_token

                self.marathi_data_loader=marathi_data_loader
                self.hindi_data_loader=hindi_data_loader
                self.magahi_data_loader=magahi_data_loader
                self.sanskrit_data_loader=sanskrit_data_loader
                self.bhojpuri_data_loader=bhojpuri_data_loader
                self.german_data_loader=german_data_loader
                self.english_data_loader=english_data_loader
                self.danish_data_loader=danish_data_loader
                self.dutch_data_loader=dutch_data_loader

                self.N=N
                self.K=K
                if N==2:
                        self.mb=2
                else:
                        self.mb=1
                
        def meta_update1(self,grads,dataloader):
                loss=0
                for _ in range(self.K*self.N):
                        x_t,y_t,sentence=dataloader.load_next()
                        loss+=self.encoder.test_train(sentence,x_t,y_t)              
                
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
                
                self.optimizer.step()

                for h in hooks:
                        h.remove()
                        
        def meta_update2(self,loss,print_epoch):
                self.optimizer.zero_grad()    
                loss.backward()
                print(str(print_epoch)+" "+str(loss.item()))
                self.optimizer.step()
                
        def train(self):
                prev_accuracy=0
                if self.N==2:
                        l=[self.marathi_data_loader,self.hindi_data_loader,self.magahi_data_loader,self.sanskrit_data_loader,
                          self.english_data_loader,self.german_data_loader,self.danish_data_loader]
                else:
                        l=[self.hindi_data_loader,self.magahi_data_loader,self.sanskrit_data_loader,
                          self.english_data_loader,self.german_data_loader]
                add=1
                for epoch in range(self.epochs):
                        
                        fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                        ls=[]
                        random.shuffle(l)
                        
                        if self.N==4:
                                data_loader1=Data_Loader(l[:self.N],self.N,self.K)
                                data_loader3=Data_Loader(l[1:],self.N,self.K)
                        else:
                                data_loader1=Data_Loader(l[:self.N],self.N,self.K)
                                data_loader2=Data_Loader(l[self.N:self.N*2],self.N,self.K,examples=2)
                                data_loader3=Data_Loader(l[2*self.N:self.N*3],self.N,self.K)
                                data_loader4=Data_Loader([l[0],l[self.N*3]],self.N,self.K,examples=2)

                        grads1,loss1=self.fast_net.train(fast_weights,data_loader1,self.N,self.K)
                        if self.N==2:
                                grads1_prime,loss1_prime=self.fast_net.train(fast_weights,data_loader2,self.N,self.K)
                                ls.append(grads1_prime)
                        ls.append(grads1)
                        grads={k: sum(d[k] for d in ls) for k in ls[0].keys()}
                        self.meta_update1(grads,data_loader1)

                        fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                        ls=[]

                        grads2,loss2=self.fast_net.train(fast_weights,data_loader3,self.N,self.K)
                        if self.N==2:
                                grads2_prime,loss2_prime=self.fast_net.train(fast_weights,data_loader4,self.N,self.K)
                                ls.append(grads2_prime)      
                        ls.append(grads2)            
                        grads={k: sum(d[k] for d in ls) for k in ls[0].keys()}
                        self.meta_update1(grads,data_loader3)
                        
                        if self.N==4:
                                loss1_prime=torch.tensor([0])
                                loss2_prime=torch.tensor([0])
                        print('epoch=',epoch+add,'training loss=',(loss1.item()+loss1_prime.item())/(self.N*self.K*self.mb))
                        print('epoch=',epoch+add+1,'training loss=',(loss2.item()+loss2_prime.item())/(self.N*self.K*self.mb))

                        add+=1
                        if (epoch+1)%5==0:
                                a,b=self.test()
                                if b>prev_accuracy:
                                        torch.save(self.encoder.state_dict(),'model_MAML_'+str(self.N)+'_way_'+str(self.K)+'_shot'+'.pth')
                                        prev_accuracy=b

        def train_Reptile(self,epsilon=0.1):
                meta_optimizer=optim.SGD(self.encoder.parameters(),lr=self.lr)
                prev_accuracy=0
                if self.N==2:
                        l=[self.marathi_data_loader,self.hindi_data_loader,self.magahi_data_loader,self.sanskrit_data_loader,
                          self.english_data_loader,self.german_data_loader,self.danish_data_loader]
                else:
                        l=[self.hindi_data_loader,self.magahi_data_loader,self.sanskrit_data_loader,
                          self.english_data_loader,self.german_data_loader]
                add=1

                for epoch in range(self.epochs):
                        random.shuffle(l)
                        if self.N==4:
                                data_loader1=Data_Loader(l[:self.N],self.N,self.K,examples=1)
                                data_loader3=Data_Loader(l[1:],self.N,self.K,examples=1)
                        else:
                                data_loader1=Data_Loader(l[:self.N],self.N,self.K,examples=1)
                                data_loader2=Data_Loader(l[self.N:self.N*2],self.N,self.K,examples=1)
                                data_loader3=Data_Loader(l[2*self.N:self.N*3],self.N,self.K,examples=1)
                                data_loader4=Data_Loader([l[0],l[self.N*3]],self.N,self.K,examples=1)

                        fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())

                        if self.N==2:
                                data_loaders1=[data_loader1,data_loader2]
                                data_loaders2=[data_loader3,data_loader4]
                                weights=[]
                                lf=0
                                for data_loader in data_loaders1:
                                        self.fast_net.clone_weights_for_test(fast_weights)
                                        inner_optimizer=optim.Adam(self.fast_net.parameters(),lr=self.lr)

                                        for _ in range(self.inner_epoch):
                                                loss=0
                                                inner_optimizer.zero_grad()
                                                for _ in range(self.N*self.K):
                                                        x,y,sentence=data_loader.load_next(reuse=True)
                                                        loss+=self.fast_net.test_train(sentence,x,y)
                                                lf+=loss.item()
                                                loss.backward()
                                                inner_optimizer.step()

                                        net_weights=OrderedDict((name,param-param_old) for ((name,param),param_old) in zip(self.fast_net.named_parameters(),fast_weights.values()))
                                        weights.append(net_weights)
                                update_weights=OrderedDict((name,(param1+param2)/2) for (name,param1,param2) in zip(weights[0].keys(),weights[0].values(),weights[1].values()))
                                for name,param in self.encoder.named_parameters():
                                        if param.grad is None:
                                                param.grad=Variable(torch.zeros(param.data.shape))
                                        param.grad.data.zero_()
                                        param.grad.data.add_(-update_weights[name])
                                meta_optimizer.step()
                                print('epoch=',epoch+add,'training loss=',lf/(self.N*self.K*self.mb*self.inner_epoch))
                                
                                lf=0
                                weights=[]
                                fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                                for data_loader in data_loaders2:
                                        self.fast_net.clone_weights_for_test(fast_weights)
                                        inner_optimizer=optim.Adam(self.fast_net.parameters(),lr=self.lr)

                                        for _ in range(self.inner_epoch):
                                                inner_optimizer.zero_grad()
                                                loss=0
                                                for _ in range(self.N*self.K):
                                                        x,y,sentence=data_loader.load_next(reuse=True)
                                                        loss+=self.fast_net.test_train(sentence,x,y)
                                                lf+=loss.item()
                                                loss.backward()
                                                inner_optimizer.step()

                                        net_weights=OrderedDict((name,param-param_old) for ((name,param),param_old) in zip(self.fast_net.named_parameters(),fast_weights.values()))
                                        weights.append(net_weights)
                                update_weights=OrderedDict((name,(param1+param2)/2) for (name,param1,param2) in zip(weights[0].keys(),weights[0].values(),weights[1].values()))
                                for name,param in self.encoder.named_parameters():
                                        if param.grad is None:
                                                param.grad=Variable(torch.zeros(param.data.shape))
                                        param.grad.data.zero_()
                                        param.grad=param.grad.data.add_(-update_weights[name])
                                meta_optimizer.step()
                                print('epoch=',epoch+1+add,'training loss=',lf/(self.N*self.K*self.mb*self.inner_epoch))

                        if self.N==4:
                                self.fast_net.clone_weights_for_test(fast_weights)
                                inner_optimizer=optim.Adam(self.fast_net.parameters(),lr=self.lr)
                                lf=0

                                for _ in range(self.inner_epoch):
                                        inner_optimizer.zero_grad()
                                        loss=0
                                        for _ in range(self.N*self.K):
                                                x,y,sentence=data_loader1.load_next(reuse=True)
                                                loss+=self.fast_net.test_train(sentence,x,y)
                                        lf+=loss.item()
                                        loss.backward()
                                        inner_optimizer.step()

                                net_weights=OrderedDict((name,param-param_old) for ((name,param),param_old) in zip(self.fast_net.named_parameters(),fast_weights.values()))
                                for name,param in self.encoder.named_parameters():
                                         if param.grad is None:
                                                param.grad=Variable(torch.zeros(param.data.shape))
                                         param.grad.data.zero_()
                                         param.grad=param.grad.data.add_(-net_weights[name])
                                meta_optimizer.step()
                                print('epoch=',epoch+add,'training loss=',lf/(self.N*self.K*self.mb*self.inner_epoch))

                                fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                                self.fast_net.clone_weights_for_test(fast_weights)
                                inner_optimizer=optim.Adam(self.fast_net.parameters(),lr=self.lr)
                                lf=0

                                for _ in range(self.inner_epoch):
                                        inner_optimizer.zero_grad()
                                        loss=0
                                        for _ in range(self.N*self.K):
                                                x,y,sentence=data_loader3.load_next(reuse=True)
                                                loss+=self.fast_net.test_train(sentence,x,y)
                                        lf+=loss.item()
                                        loss.backward()
                                        inner_optimizer.step()

                                net_weights=OrderedDict((name,param-param_old) for ((name,param),param_old) in zip(self.fast_net.named_parameters(),fast_weights.values()))
                                for name,param in self.encoder.named_parameters():
                                         if param.grad is None:
                                                param.grad=Variable(torch.zeros(param.data.shape))
                                         param.grad.data.zero_()
                                         param.grad=param.grad.data.add_(-net_weights[name])
                                meta_optimizer.step()
                                print('epoch=',epoch+1+add,'training loss=',lf/(self.N*self.K*self.mb*self.inner_epoch))
                        
                        add+=1
                        if (epoch+1)%5==0:
                                a,b=self.test()
                                if b>prev_accuracy:
                                        torch.save(self.encoder.state_dict(),'model_Reptile_'+str(self.N)+'_way_'+str(self.K)+'_shot'+'.pth')
                                        prev_accuracy=b
                        
        def test(self,t=2,num=40):
                accuracy_final1=0
                accuracy_final2=0

                for _ in range(num):
                        fast_weights=OrderedDict((name,param) for (name,param) in self.encoder.named_parameters())
                        self.fast_net.clone_weights_for_test(fast_weights)
                        train_optimizer=optim.Adam(self.fast_net.parameters(),lr=self.lr)
                        
                        if self.N==2:
                                loaders=[self.bhojpuri_data_loader,self.dutch_data_loader]
                        else:
                                loaders=[self.bhojpuri_data_loader,self.dutch_data_loader,self.danish_data_loader,self.marathi_data_loader]
                        random.shuffle(loaders)
                        data_loader=Data_Loader(loaders,self.N,K=self.K,examples=2)

                        for _ in range(self.inner_epoch):
                                train_optimizer.zero_grad()
                                loss=0
                                for _ in range(self.N*self.K):
                                        x_test,y_test,sentence=data_loader.load_next(reuse=True)
                                        loss+=self.fast_net.test_train(sentence,x_test,y_test)
                                loss.backward()
                                train_optimizer.step()
                        
                        data_loader.set_counter()
                        a,b,c=0,0,0
                        for _ in range(t):
                                x_test,y_test,sentence1=data_loader.load_next()
                                score,outputprime=self.fast_net.forward(x_test,sentence1)
                                
                                j,count=0,0
                                for i in range(len(sentence1)):
                                        c+=1
                                        if outputprime[j]==y_test[j]:
                                                count+=1
                                                b+=1
                                        j+=1
                                
                                accuracy=100*count/j
                                a+=accuracy
                                
                        accuracy_final1+=a/t
                        accuracy_final2+=b*100/c
                print('validation accuracy over sentences=',accuracy_final1/num,'validation accuracy over tags=',accuracy_final2/num)

                return accuracy_final1/num,accuracy_final2/num

        def save_checkpoint(self,epoch,train_type):
                save_model={'epochs_left':epoch,'model':self.encoder.state_dict()}   
                torch.save(save_model,'checkpoint_model_'+train_type+'_'+str(self.N)+'_way_'+str(self.K)+'_shot'+'.pth')                 