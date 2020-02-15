import pyconll
import os
import torch
from functions import preprocess_2

def load_sentences():
        hindi_train=os.getcwd()+'/hi_hdtb-ud-train.conllu'
        marathi_train=os.getcwd()+'/mr_ufal-ud-train.conllu'
        marathi_test=os.getcwd()+'/mr_ufal-ud-test.conllu'
        hindi_test=os.getcwd()+'/hi_hdtb-ud-test.conllu'
        
        sentences_marathi_train=preprocess_2(pyconll.load_from_file(marathi_train))
        sentences_marathi_test=preprocess_2(pyconll.load_from_file(marathi_test))
        sentences_hindi_train=pyconll.load_from_file(hindi_train)
        sentences_hindi_test=pyconll.load_from_file(hindi_test)
        
        return  sentences_marathi_train, sentences_marathi_test, sentences_hindi_train, sentences_hindi_test


def get_sentences(sentences_train,sentences_test,tags,max_len):
        sentences_for_test=[]
        sentences_for_train=[]
        tags_for_test=[]
        tags_for_train=[]
        
        for sentence in sentences_train:
                k=[]
                t=[]
                for token in sentence:
                        if token.form is not None:
                                k.append(token.form)
                                t.append(tags[token.upos])
                for _ in range(len(k),max_len):
                        k.append('EOS')
                        t.append(tags['X'])
                sentences_for_train.append(k)
                tags_for_train.append(t)

        for sentence in sentences_test:
                k=[]
                t=[]
                for token in sentence:
                        if token.form is not None:
                                k.append(token.form)
                                t.append(tags[token.upos])
                for _ in range(len(k),max_len):
                        k.append('EOS')
                        t.append(tags['X'])
                sentences_for_test.append(k)
                tags_for_test.append(t)
                
        return sentences_for_train,sentences_for_test,tags_for_train,tags_for_test
            
def get_tokens(sentences):
        s=set()
        for sentence in sentences:
                for token in sentence:
                        s.add(token.upos)
        s.add('START')
        s.add('END')
        s=list(s)
        dict={}
        dict1={}
        for i in range(len(s)):
                dict[s[i]]=i
                dict1[i]=s[i]
        
        return dict,dict1,len(s)
        

class DataLoader(object):
        def __init__(self,train_sentences,test_sentences,train_tags,test_tags,max_len,model):
                self.train=train_sentences
                self.test=test_sentences
                self.train_number=0
                self.test_number=0
                self.max_len=max_len
                self.model=model
                self.train_tags=train_tags
                self.test_tags=test_tags
                
        def load_next(self):
                sentence=self.train[self.train_number]
                tags=self.train_tags[self.train_number]
                l=[]
                
                for token in sentence:
                        l.append(self.model[token])
                        
                embedding=torch.tensor(l).view(1,self.max_len,-1)  #.cuda()
                self.train_number=(self.train_number+1)%len(self.train)
                tags=torch.tensor(tags)  #.cuda()
                
                return embedding,tags
            
        def load_next_test(self):
                sentence=self.test[self.test_number]
                tags=self.test_tags[self.test_number]
                l=[]
                
                for token in sentence:
                        l.append(self.model[token])
                        
                embedding=torch.tensor(l).view(1,len(sentence),-1)
                self.test_number=(self.test_number+1)%len(self.test)
                tags=torch.tensor(tags)
                
                return embedding,tags,sentence