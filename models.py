import torch.nn as nn

class POSTagger(nn.Module):
    
        def __init__(self,h_size,n_tokens):
                super(POSTagger,self).__init__()
                self.lstm=nn.LSTM(h_size,h_size,num_layers=1,bidirectional=True)
                self.Dense=nn.Linear(h_size*2,n_tokens)     
                
        def forward(self,input,weights=None,hidden=None):
                
                output,hidden=self.lstm(input,hidden)
                output=self.Dense(output)
                output=output.squeeze()
                        
                return output,hidden


class Word:
        def __init__(self,data,data_test):
                self.n_words=0
                self.n_tokens=16
                self.word_to_int={}
                self.int_to_word={}
                self.data=data
                self.token_to_int={}
                self.int_to_token={}
                self.data_test=data_test
                
        def addWords(self):
                s=set()
                #s.add('PAD')
                k=set()
                for sentence in self.data:
                        for token in sentence:
                                if token.form is not None:
                                        s.add(token.form)
                                k.add(token.upos)
                            
                for sentence in self.data_test:
                        for token in sentence:
                                if token.form is not None:
                                        s.add(token.form)
                                k.add(token.upos)
                s=list(s)
                k=list(k)
                self.n_words=len(s)
                self.create_dict(s,k)
                
        def create_dict(self,s,k):
                for i in range(len(s)):
                        self.int_to_word[i]=s[i]
                        self.word_to_int[s[i]]=i
                        
                for i in range(len(k)):
                        self.int_to_token[i]=k[i]
                        self.token_to_int[k[i]]=i