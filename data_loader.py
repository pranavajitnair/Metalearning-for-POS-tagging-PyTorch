import pyconll
import os
import torch
from functions import preprocess_2

def load_sentences():
        hindi_train=os.getcwd()+'/hi_hdtb-ud-train.conllu'
        marathi_train=os.getcwd()+'/mr_ufal-ud-train.conllu'
        marathi_test=os.getcwd()+'/mr_ufal-ud-test.conllu'
        marathi_dev=os.getcwd()+'/mr_ufal-ud-dev.conllu'
        bhojpuri_train=os.getcwd()+'/bho_bhtb-ud-test.conllu'
        magahi_train=os.getcwd()+'/mag_mgtb-ud-test.conllu'
        sanskrit_train=os.getcwd()+'/sa_ufal-ud-test.conllu'
        english_train=os.getcwd()+'/en_gum-ud-dev.conllu'
        german_train=os.getcwd()+'/de_hdt-ud-dev.conllu'
        dutch_train=os.getcwd()+'/nl_lassysmall-ud-test.conllu'
        danish_train=os.getcwd()+'/da_ddt-ud-test.conllu'
        
        sentences_marathi_train=preprocess_2(pyconll.load_from_file(marathi_train))
        sentences_marathi_test=preprocess_2(pyconll.load_from_file(marathi_test))
        sentences_marathi_dev=preprocess_2(pyconll.load_from_file(marathi_dev))
        sentences_hindi_train=preprocess_2(pyconll.load_from_file(hindi_train))
        sentences_magahi_train=preprocess_2(pyconll.load_from_file(bhojpuri_train))
        sentences_bhojpuri_train=preprocess_2(pyconll.load_from_file(magahi_train))
        sentences_sanskrit_train=preprocess_2(pyconll.load_from_file(sanskrit_train))
        sentences_english_train=preprocess_2(pyconll.load_from_file(english_train))
        sentences_german_train=preprocess_2(pyconll.load_from_file(german_train))
        sentences_dutch_train=preprocess_2(pyconll.load_from_file(dutch_train))
        sentences_danish_train=preprocess_2(pyconll.load_from_file(danish_train))
        
        return sentences_sanskrit_train,sentences_marathi_train,sentences_marathi_test,sentences_marathi_dev,sentences_hindi_train,sentences_bhojpuri_train,sentences_magahi_train,sentences_english_train,sentences_german_train,sentences_dutch_train,sentences_danish_train


def get_sentences(sentences_train,sentences_test,tags,max_len):
        sentences_for_train=[]
        tags_for_train=[]
        
        for sentence in sentences_train:
                k=[]
                t=[]
                for token in sentence:
                        if token.form is not None:
                                if token.form=='।':
                                        k.append('.')
                                else:
                                        k.append(token.form)
                                t.append(tags[token.upos])
#                k.append('EOS')
#                t.append(tags['X'])
#                for _ in range(len(k),max_len):
#                        k.append('EOS')
#                        t.append(tags['X'])
                sentences_for_train.append(k)
                tags_for_train.append(t)

        return sentences_for_train,tags_for_train

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
        
def get_characters(sentences):
        s=set()
        for sentence in sentences:
                for word in sentence:
                        for character in word:
                                s.add(character)
                                
        s=list(s)
        
        dict={}
        for i in range(len(s)):
                dict[s[i]]=i

        dict['pad']=len(s)
                
        return dict,len(s)+1


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
                        
                embedding=torch.tensor(l).view(1,len(sentence),-1) #.cuda()
                self.train_number=(self.train_number+1)%len(self.train)
                tags=torch.tensor(tags) #.cuda()
                
                return embedding,tags,sentence
            
        def load_next_test(self):
                sentence=self.test[self.test_number]
                tags=self.test_tags[self.test_number]
                l=[]
                
                for token in sentence:
                        l.append(self.model[token])
                        
                embedding=torch.tensor(l).view(1,len(sentence),-1) #.cuda()
                self.test_number=(self.test_number+1)%len(self.test)
                tags=torch.tensor(tags) #.cuda()
                
                return embedding,tags,sentence
            
            
class Data_Loader(object):
        def __init__(self,dataloaders,N,K,examples=3):
                self.counter=0
                self.K=K
                self.data=[]
                self.N=N

                for _ in range(examples):
                        for dataloader in dataloaders:
                                for i in range(self.K):
                                        self.data.append(dataloader.load_next())

        def load_next(self,reuse=False):
                data=self.data[self.counter]
                if reuse:
                        self.counter=(self.counter+1)%(self.N*self.K)
                else:
                        self.counter+=1

                return data

        def set_counter(self):
                self.counter=self.N*self.K