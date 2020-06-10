import torch
import torch.nn as nn

from collections import OrderedDict

            
class CRF_BiLSTM(nn.Module):
        def __init__(self,epochs,h_size,n_tokens,token_dict,char_dict,n_chars):
                super(CRF_BiLSTM,self).__init__()
                
                self.h_size=h_size
                self.n_tokens=n_tokens
               
                self.epochs=epochs
                self.start_token='START'
                self.end_token='END'
                self.token_dict=token_dict
                self.char_dict=char_dict
                self.n_chars=n_chars
                self.char_dim=17
                
                self.embeddings=nn.Embedding(self.n_chars,self.char_dim)
                nn.init.xavier_uniform_(self.embeddings.weight)
                
                self.transitions=nn.Parameter(torch.randn(self.n_tokens,self.n_tokens))
                nn.init.xavier_uniform_(self.transitions.data)
                
                self.lstm=nn.LSTM(h_size,h_size,num_layers=1,bidirectional=True,batch_first=True,dropout=0.2)
                
                for name,weight in self.lstm.named_parameters():
                        if 'weight' in name:
                                nn.init.xavier_uniform_(weight)        
                
                self.Dense1=nn.Linear(h_size*4,self.n_tokens)
                nn.init.xavier_uniform_(self.Dense1.weight)
                
                self.transitions.data[self.token_dict[self.start_token], :]=-10000.0
                self.transitions.data[:,self.token_dict[self.end_token]]=-10000.0

                self.conv1=nn.Conv1d(self.char_dim,64,2)
                self.conv2=nn.Conv1d(self.char_dim,64,2)
                self.conv3=nn.Conv1d(self.char_dim,64,3)
                self.conv4=nn.Conv1d(self.char_dim,64,3)                
                
        def argmax(vec):
                _, idx=torch.max(vec,1)
                
                return idx.item()
                         
        def get_lstm_feats(self,char_list,sentence,weights):
            
                if weights:
                        self.load_state_dict(weights)

                char_list=torch.tensor(char_list)
                char_embeds=self.embeddings(char_list).view(sentence.shape[1],-1,self.char_dim).transpose(1,2)

                o1=self.conv1(char_embeds)
                o2=self.conv2(char_embeds)
                o3=self.conv3(char_embeds)
                o4=self.conv4(char_embeds)

                o1,_=torch.max(o1,dim=-1)
                o2,_=torch.max(o2,dim=-1)
                o3,_=torch.max(o3,dim=-1)
                o4,_=torch.max(o4,dim=-1)

                l=torch.cat([o1,o2,o3,o4],dim=-1).unsqueeze(0)
                
                output,hidden=self.lstm(sentence,None)
                output=torch.cat([output,sentence,l],dim=-1)
                        
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
                tags = torch.cat([torch.tensor([self.token_dict[self.start_token]], dtype=torch.long), tags]) #.cuda() #.cuda()
                for i, feat in enumerate(feats):
                        score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
                score = score + self.transitions[self.token_dict[self.end_token], tags[-1]]
                
                return score
                
        def forward_prop(self,feats):
            
                init_alphas=torch.full((1,self.n_tokens),-10000.) #.cuda()
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
                
        def neg_log_likelihood(self,char_list,sentence,tags,weights=None):
                feats=self.get_lstm_feats(char_list,sentence,weights)
                forward_score=self.forward_prop(feats)
                gold_score=self.score_sentence(feats,tags)
                
                return forward_score-gold_score
            
        def viterbi_decode(self,feats):
                backpointers = []
                init_vvars = torch.full((1, self.n_tokens), -10000.) #.cuda()
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
            
                        forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1) #.cuda()
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
            
        def train(self,weights,data_loader,N,K,return_weights=False,return_grads=False):
                weights_clone=self.clone_weights(weights)
                self.load_state_dict(weights_clone)
                
                for _ in range(self.epochs):
                        loss=0
                        for _ in range(N*K):
                                 sentence,tags,sentence_text=data_loader.load_next(reuse=True)
                                 char_list=self.get_characters(sentence_text)
                                 loss+=self.neg_log_likelihood(char_list,sentence,tags) #,weights_clone
                        
                        grads=torch.autograd.grad(loss,self.parameters(),create_graph=True)
                        
                        weights_clone=OrderedDict((name, param - 0.01*grad) for ((name, param), grad) in zip(weights_clone.items(),grads ))

                if return_weights:
  
                        return weights_clone,loss.item()

                if return_grads:
                        meta_weights=OrderedDict((name,grad) for ((name,param),grad) in zip(weights_clone.items(),grads ))
                        return meta_weights,loss.item()

                loss=0
                data_loader.set_counter()
                for _ in range(N*K):
                        sentence,tags,sentence_text=data_loader.load_next()
                        char_list=self.get_characters(sentence_text)
                        
                        loss+=self.neg_log_likelihood(char_list,sentence,tags) #,weights_clone
                
                grads=torch.autograd.grad(loss,self.parameters(),create_graph=True)
                meta_grads={name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
                        
                return meta_grads,loss
            
        def forward(self,sentence,sentence_text):
            
                char_list=self.get_characters(sentence_text)
                lstm_feats=self.get_lstm_feats(char_list,sentence,None)
                score,tag_seq=self.viterbi_decode(lstm_feats)
                
                return score,tag_seq
            
        def test_train(self,sentence_text,sentence,tags):
                char_list=self.get_characters(sentence_text)
                
                loss=self.neg_log_likelihood(char_list,sentence,tags,None)
                
                return loss
            
        def get_characters(self,sentence):
                max1=0
                for word in sentence:
                        max1=max(max1,len(word))
                max1=max(max1,5)

                s=[]
                # s.append(-1)
                for word in sentence:
                        char_list=[]
                        for character in word:
                                char_list.append(self.char_dict[character])
                                # s.append(self.char_dict[character])
                        for _ in range(max1-len(word)):
                                char_list.append(self.char_dict['pad'])
                        s.append(char_list)
                        # s.append(-1)
                
                return s
            
        def clone_weights(self,weights):
                weights_clone=OrderedDict()
                for name,_ in weights.items():
                        weights_clone[name]=weights[name].clone()

                return weights_clone

        def clone_weights_for_test(self,weights):
                weights_clone=self.clone_weights(weights)
                self.load_state_dict(weights_clone)