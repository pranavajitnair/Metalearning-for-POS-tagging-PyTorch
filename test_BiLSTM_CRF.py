from data_loader import load_sentences,get_sentences,get_tokens,DataLoader
import gensim.models as gs
import torch
import torch.nn as nn
import torch.optim as optim


class CRF_BiLSTM(nn.Module):
        def __init__(self,epochs,h_size,n_tokens,data_loader,token_dict):
                super(CRF_BiLSTM,self).__init__()
                
                self.h_size=h_size
                self.n_tokens=n_tokens
                self.data_loader=data_loader
                self.epochs=epochs
                self.start_token='START'
                self.end_token='END'
                self.token_dict=token_dict
                
                self.transitions=nn.Parameter(torch.randn(self.n_tokens,self.n_tokens))
                self.lstm=nn.LSTM(h_size,h_size,num_layers=1,bidirectional=True)
                self.Dense1=nn.Linear(h_size*2,self.n_tokens)
                
                self.transitions.data[self.token_dict[self.start_token], :]=-10000.0
                self.transitions.data[:,self.token_dict[self.end_token]]=-10000.0
                
                self.optimizer=optim.Adam(self.parameters(),lr=0.01)
                
        def argmax(vec):
                _, idx=torch.max(vec,1)
                
                return idx.item()
                         
        def get_lstm_feats(self,sentence):
            
                output,hidden=self.lstm(sentence,None)
                output=self.Dense1(output)
                output=output.squeeze()
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
                
        def neg_log_likelihood(self,sentence,tags):
                feats=self.get_lstm_feats(sentence)
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
            
        def train(self):
                self.optimizer.zero_grad()
                sentence,tags=self.data_loader.load_next()
                loss=self.neg_log_likelihood(sentence,tags)
                print(loss.item())
                loss.backward()
               
                self.optimizer.step()
               
            
        def forward(self,sentence):
                lstm_feats=self.get_lstm_feats(sentence)
                score,tag_seq=self.viterbi_decode(lstm_feats)
                
                return score,tag_seq
            

max_len=116
hidden_size=1024
epochs=400

marathi_train,marathi_test,hindi_train,hindi_test=load_sentences()
tokens_dict,dict_token,n_tokens=get_tokens(marathi_train)

marathi_train,marathi_test,marathi_train_tags,marathi_test_tags=get_sentences(marathi_train,marathi_test,tokens_dict,max_len)

model_marathi=gs.Word2Vec(marathi_test+marathi_train,min_count=1,size=hidden_size)

marathi_data_loader=DataLoader(marathi_train,marathi_test,marathi_train_tags,marathi_test_tags,max_len,model_marathi)

model=CRF_BiLSTM(1,hidden_size,n_tokens,marathi_data_loader,tokens_dict)

for i in range(epochs):
        print(i)
        model.train()
        
for _ in range(30):
    
        sentence,tags,sentence_text=model.data_loader.load_next_test()
        score,tag_seq=model.forward(sentence)
                
        count=0
        j=0
        s1=''
        s2=''
        s3=''
        
        while sentence_text[j]!='EOS':
                s1+=sentence_text[j]+' '
                s2+=dict_token[tag_seq[j]]+' '
                s3+=dict_token[int(tags[j])]+' '
                if tag_seq[j]==tags[j]:
                        count+=1
                j+=1
        
        print(100*(count/(j)))
        print(s1)
        print(s2)
        print(s3)
                
        
        
