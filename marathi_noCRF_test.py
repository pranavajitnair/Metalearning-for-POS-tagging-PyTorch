from data_loader import load_sentences,get_sentences,DataLoader
import gensim.models as gs
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
        def __init__(self,h_size,n_tokens,data_loader,tokens_dict,max_len):
                super(Model,self).__init__()
                
                self.lstm=nn.LSTM(h_size,h_size,bidirectional=False)
                self.Dense=nn.Linear(h_size,n_tokens)
                
                self.lossFunction=nn.CrossEntropyLoss()
                self.optimizer=optim.Adam(self.parameters(),lr=0.01)
                
                self.data_loader=data_loader
                self.index_to_token=tokens_dict
                self.max_len=max_len
                self.n_tokens=n_tokens
                
        def forward(self,input):
                output,hidden=self.lstm(input,None)
                output=self.Dense(output)
                output=output.squeeze()
                
                return output
        
        def train(self):
                input,tags=self.data_loader.load_next()
                
                self.optimizer.zero_grad()
                
                output=self.forward(input)
                loss=self.lossFunction(output,tags)
                print(loss.item())
                loss.backward()
                self.optimizer.step()
                
        def test(self):
                input,y_test,sentence1=self.data_loader.load_next_test()
                input=self.forward(input)
                output=F.softmax(input)
                
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
                print('Accuracy= '+str(100*(count/(j)))+'%')
                print('')
                
                
def get_tokens(sentences):
        s=set()
        for sentence in sentences:
                for token in sentence:
                        s.add(token.upos)
        s=list(s)
        dict={}
        dict1={}
        for i in range(len(s)):
                dict[s[i]]=i
                dict1[i]=s[i]
        
        return dict,dict1,len(s)
        
        
max_len=116
hidden_size=1024
epochs=400

marathi_train,marathi_test,hindi_train,hindi_test=load_sentences()
tokens_dict,dict_token,n_tokens=get_tokens(marathi_train)

marathi_train,marathi_test,marathi_train_tags,marathi_test_tags=get_sentences(marathi_train,marathi_test,tokens_dict,max_len)

model_marathi=gs.Word2Vec(marathi_test+marathi_train,min_count=1,size=hidden_size)

marathi_data_loader=DataLoader(marathi_train,marathi_test,marathi_train_tags,marathi_test_tags,max_len,model_marathi)

model=Model(hidden_size,n_tokens,marathi_data_loader,dict_token,max_len)

for _ in range(epochs):
        model.train()
        
for _ in range(30):
        model.test()