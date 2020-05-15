from data_loader import load_sentences,get_sentences,get_tokens,DataLoader,get_characters
import gensim.models as gs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#class MultiheadAttention(nn.Module):
#        def __init__(self,h_size,embed_size,n_heads):
#                super(MultiheadAttention,self).__init__()
#                self.h_size=h_size
#                self.embed_size=embed_size
#                
#                self.q=nn.Linear(self.embed_size,self.h_size)
#                self.v=nn.Linear(self.embed_size,self.h_size)
#                self.k=nn.Linear(self.embed_size,self.h_size)
#                
#                self.n_heads=n_heads
#                self.Dense=nn.Linear(self.h_size,self.embed_size)
#                
#        def forward(self,sentence):
#                seq_len=sentence.shape[0]
#                
#                qq=self.q(sentence)
#                kk=self.k(sentence)
#                vv=self.v(sentence)
#                
#                qq=qq.view(seq_len,self.n_heads,self.h_size//self.n_heads)
#                kk=kk.view(seq_len,self.n_heads,self.h_size//self.n_heads)
#                vv=vv.view(seq_len,self.n_heads,self.h_size//self.n_heads)
#                
#                qq=qq.transpose(0,1)
#                kk=kk.transpose(0,1)
#                vv=vv.transpose(0,1)
#                
#                at=torch.bmm(qq,kk.transpose(1,2))/math.sqrt(self.h_size)
#                at=F.softmax(at,dim=2)
#                
#                output=torch.bmm(at,vv)
#                output=output.transpose(0,1)
#                output=output.contiguous().view(seq_len,-1)
#                
#                return self.Dense(output)
#            


class CRF_BiLSTM(nn.Module):
        def __init__(self,epochs,h_size,n_tokens,data_loader,token_dict,char_dict,n_chars,test_size):
                super(CRF_BiLSTM,self).__init__()
                
                self.h_size=h_size
                self.n_tokens=n_tokens
                self.data_loader=data_loader
                self.epochs=epochs
                self.start_token='START'
                self.end_token='END'
                self.token_dict=token_dict
                self.char_dict=char_dict
                self.n_chars=n_chars
                self.test_size=test_size
                
#                self.softmax_weights=nn.Parameter(torch.randn(3))
                
                self.embeddings=nn.Embedding(self.n_chars,h_size67)
                nn.init.xavier_uniform_(self.embeddings.weight)
                
                self.transitions=nn.Parameter(torch.randn(self.n_tokens,self.n_tokens))
                nn.init.xavier_uniform_(self.transitions.data)
                
                self.lstm=nn.LSTM(h_size,h_size,num_layers=1,bidirectional=True)
                
                for name,weight in self.lstm.named_parameters():
                        if 'weight' in name:
                                nn.init.xavier_uniform_(weight)        
#                
                self.Dense1=nn.Linear(h_size*2,self.n_tokens)
                nn.init.xavier_uniform_(self.Dense1.weight)
                
                self.transitions.data[self.token_dict[self.start_token], :]=-10000.0
                self.transitions.data[:,self.token_dict[self.end_token]]=-10000.0
                
                self.optimizer=optim.Adam(self.parameters(),lr=0.01)
                
#                self.Dense1=nn.Linear(h_size,h_size)    
#                self.Dense2=nn.Linear(h_size,self.n_tokens)
#               
#                self.attention1=MultiheadAttention(512,256,4)
#                self.attention2=MultiheadAttention(512,256,4)
#                self.attention3=MultiheadAttention(512,256,4)
#                self.attention4=MultiheadAttention(512,256,4)
               
        def argmax(vec):
                _, idx=torch.max(vec,1)
                
                return idx.item()
                         
        def get_lstm_feats(self,sentence,char_list):
#                sentence=sentence.squeeze(0)
#                positional=torch.zeros(sentence.shape,requires_grad=False)
#                
#                for i in range(sentence.shape[0]):
#                        for j in range(0,sentence.shape[1],2):
#                                positional[i][j]=math.sin(i/(10000**(2*(j)/sentence.shape[1])))
#                                positional[i][j+1]=math.cos(i/(10000**(2*(j+1)/sentence.shape[1])))
#                                
#                sentence+=positional
#
#                output=self.attention1(sentence)
#                output=self.attention2(output)
#                output=self.attention3(output)
#                output=self.attention4(output)
#                
#                output=self.Dense1(output)
#                output=self.Dense2(output)
#                
#                return output
            
                l=[]
                sumlist=torch.ones((1,self.h_size))
                hidden=None
                for char_number in char_list:
                        if char_number==-1:
                                sumlist=F.relu(sumlist) #test
                                l.append(sumlist)
                                sumlist=torch.ones((1,self.h_size))  #256 good
                                hidden=None
                        else:
                                embedding=(self.embeddings(torch.tensor(char_number)))**2
                                sumlist*=embedding
                
                l=l[1:]
                
                for i in range(len(l)):
                        em=l[i].squeeze()
                        sentence[0][i]+=em
                
                output,hidden=self.lstm(sentence,None)
                
                
                output=self.Dense1(output)
                output=output.squeeze()
                output=output.view(-1,self.n_tokens)
                
                return output
                
        def log_sum_exp(self,vec):
                _, idx=torch.max(vec,1)
                max_score=vec[0,idx.item()]
                max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
                
                return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
            
        def score_sentence(self,feats,tags):
                score = torch.zeros(1) #.cuda()
                tags = torch.cat([torch.tensor([self.token_dict[self.start_token]], dtype=torch.long), tags])   #.cuda()
                for i, feat in enumerate(feats):
                        score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
                score = score + self.transitions[self.token_dict[self.end_token], tags[-1]]
                
                return score
                
        def forward_prop(self,feats):
            
                init_alphas=torch.full((1,self.n_tokens),-10000.)  #.cuda()
                init_alphas[0][self.token_dict[self.start_token]]=0.
                forward_var = init_alphas
                
                for feat in feats:
                         alphas_t=[]
                    
                         for next_tag in range(self.n_tokens):
                                 emit_score=feat[next_tag].view(1,-1).expand(1,self.n_tokens)
                                 trans_score=self.transitions[next_tag].view(1, -1)
                                 next_tag_var=forward_var+trans_score+emit_score
                                 alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
                         forward_var = torch.cat(alphas_t).view(1, -1)
                         
                terminal_var=forward_var+self.transitions[self.token_dict[self.end_token]]
                alpha=self.log_sum_exp(terminal_var)
                
                return alpha
                
        def neg_log_likelihood(self,sentence,tags,char_list):
                feats=self.get_lstm_feats(sentence,char_list)
                forward_score=self.forward_prop(feats)
                gold_score=self.score_sentence(feats,tags)
                
                return forward_score-gold_score
            
        def viterbi_decode(self,feats):
                backpointers = []
                init_vvars = torch.full((1, self.n_tokens), -10000.)  #.cuda()
                init_vvars[0][self.token_dict[self.start_token]] = 0
                forward_var = init_vvars
                for feat in feats:
                        bptrs_t = []  
                        viterbivars_t = []  
            
                        for next_tag in range(self.n_tokens):
                                next_tag_var=forward_var+self.transitions[next_tag]
                                _, idx=torch.max(next_tag_var,1)
                                best_tag_id=idx.item()
                                bptrs_t.append(best_tag_id)
                                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            
                        forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                        backpointers.append(bptrs_t)
        
                terminal_var=forward_var + self.transitions[self.token_dict[self.end_token]]
                _, idx=torch.max(terminal_var,1)
                best_tag_id=idx.item()
                path_score=terminal_var[0][best_tag_id]
                best_path=[best_tag_id]
                
                for bptrs_t in reversed(backpointers):
                        
                        best_tag_id = bptrs_t[best_tag_id]
                        best_path.append(best_tag_id)
                        
                start = best_path.pop()
                assert start == self.token_dict[self.start_token]  
                best_path.reverse()
                
                return path_score, best_path
            
        def train_epoch(self,iterations):
                self.optimizer.zero_grad()
                loss=0
                
                for _ in range(iterations):
                        sentence,tags,sentence_text=self.data_loader.load_next()
                        char_list=self.get_characters(sentence_text) 
                        
                        loss+=self.neg_log_likelihood(sentence,tags,char_list)
                        
                loss_val=loss.item()
                loss.backward()
               
                self.optimizer.step()
                
                return loss_val/iterations
               
            
        def forward(self,sentence,sentence_text):
                char_list=self.get_characters(sentence_text)            
                
                lstm_feats=self.get_lstm_feats(sentence,char_list)
                score,tag_seq=self.viterbi_decode(lstm_feats)
                
                return score,tag_seq
            
        def get_characters(self,sentence):
                s=[]
                s.append(-1)
                for word in sentence:
                        for character in word:
                                s.append(self.char_dict[character])
                        s.append(-1)
                
                return s
            
        def test(self):
                a=0
                b=0
                c=0
                for _ in range(self.test_size):
                    
                        sentence,tags,sentence_text=self.data_loader.load_next_test()
                        score,tag_seq=model.forward(sentence,sentence_text)
                                
                        count=0
                        j=0
#                        s1=''
#                        s2=''
#                        s3=''nn
                        
                        for _ in range(len(sentence_text)):
                                c+=1
#                                s1+=sentence_text[j]+' '
#                                s2+=dict_token[tag_seq[j]]+' '
#                                s3+=dict_token[int(tags[j])]+' '
                                if tag_seq[j]==tags[j]:
                                        count+=1
                                        b+=1
                                j+=1
                                
                        accuracy=100*(count/(j))
                        a+=accuracy
#                        print(accuracy)
#                        print(s1)
#                        print(s2)
#                        print(s3)
                        
#                print()
                print('test accuracy1=',a/self.test_size)
                print('test accuracy2',100*b/c)
            

max_len=116
hidden_size=256
epochs=200
iterations=373
test_size=47

marathi_train,marathi_test,marathi_trai,marathi_tes=load_sentences()
tokens_dict,dict_token,n_tokens=get_tokens(marathi_train)

marathi_train,marathi_test,marathi_train_tags,marathi_test_tags=get_sentences(marathi_train,marathi_test,tokens_dict,max_len)

model_marathi=gs.Word2Vec(marathi_test+marathi_train,min_count=1,size=hidden_size)

char_dict,n_chars=get_characters(marathi_train)

marathi_data_loader=DataLoader(marathi_train,marathi_test,marathi_train_tags,marathi_test_tags,max_len,model_marathi)

model=CRF_BiLSTM(1,hidden_size,n_tokens,marathi_data_loader,tokens_dict,char_dict,n_chars,test_size)
model.train()

for i in range(epochs):
        loss=model.train_epoch(iterations)
        
        if i%20==0 and i!=0:
                model.test()
                
        print('epoch=',i+1,'training loss=',loss)

model.eval()     
model.test()