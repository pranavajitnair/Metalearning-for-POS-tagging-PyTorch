import torch
import torch.nn as nn

import gensim.models as gs
import argparse

from data_loader import DataLoader,get_tokens,get_sentences,load_sentences,get_characters
from maml import MetaLearn

def main(args):
                
        lossFunction=nn.CrossEntropyLoss()
        
        hidden_size=args.hidden_size
        epsilon=args.epsilon
        training_mode=args.training_mode
        learning_rate=args.learning_rate
        
        epochs=args.epochs
        K=args.K_shot_learning
        N=args.N_way_learning
        inner_epoch=args.inner_gradient_update
        max_len=116
        
        sanskrit_train,marathi_train,marathi_test,marathi_dev,hindi_train,bhojpuri_train,magahi_train,english_train,german_train,dutch_train,danish_train=load_sentences()
        tokens_dict,dict_token,n_tokens=get_tokens(bhojpuri_train)
        
        marathi,marathi_tags=get_sentences(marathi_train,None,tokens_dict,max_len)
        marathi_d,marathi_tags_d=get_sentences(marathi_dev,None,tokens_dict,max_len)
        marathi_t,marathi_tags_t=get_sentences(marathi_test,None,tokens_dict,max_len)
        hindi,hindi_tags=get_sentences(hindi_train,None,tokens_dict,max_len)
        bhojpuri,bhojpuri_tags=get_sentences(bhojpuri_train,None,tokens_dict,max_len)
        magahi,magahi_tags=get_sentences(magahi_train,None,tokens_dict,max_len)
        sanskrit,sanskrit_tags=get_sentences(sanskrit_train,None,tokens_dict,max_len)
        marathi=marathi+marathi_d+marathi_t
        marathi_tags=marathi_tags+marathi_tags_d+marathi_tags_t
        
        english,english_tags=get_sentences(dutch_train,None,tokens_dict,max_len)
        dutch,dutch_tags=get_sentences(dutch_train,None,tokens_dict,max_len)
        danish,danish_tags=get_sentences(danish_train,None,tokens_dict,max_len)
        german,german_tags=get_sentences(german_train,None,tokens_dict,max_len)
        
        model_hindi=gs.Word2Vec(hindi,min_count=1,size=hidden_size)        
        model_marathi=gs.Word2Vec(marathi,min_count=1,size=hidden_size)   
        model_sanskrit=gs.Word2Vec(sanskrit,min_count=1,size=hidden_size) 
        model_bhojpuri=gs.Word2Vec(bhojpuri,min_count=1,size=hidden_size)   
        model_magahi=gs.Word2Vec(magahi,min_count=1,size=hidden_size)       
        model_german=gs.Word2Vec(german,min_count=1,size=hidden_size)  
        model_english=gs.Word2Vec(english,min_count=1,size=hidden_size)
        model_dutch=gs.Word2Vec(dutch,min_count=1,size=hidden_size)  
        model_danish=gs.Word2Vec(danish,min_count=1,size=hidden_size)  
        
        char_dict,n_chars=get_characters(marathi+hindi+bhojpuri+sanskrit+magahi+english+dutch+danish+german)
        
        hindi_data_loader=DataLoader(hindi,None,hindi_tags,None,max_len,model_hindi)
        marathi_data_loader=DataLoader(marathi,None,marathi_tags,None,max_len,model_marathi)
        sanskrit_data_loader=DataLoader(sanskrit,None,sanskrit_tags,None,max_len,model_sanskrit)
        bhojpuri_data_loader=DataLoader(bhojpuri,None,bhojpuri_tags,None,max_len,model_bhojpuri)
        magahi_data_loader=DataLoader(magahi,None,magahi_tags,None,max_len,model_magahi)
        english_data_loader=DataLoader(english,None,english_tags,None,max_len,model_english)
        german_data_loader=DataLoader(german,None,german_tags,None,max_len,model_german)
        danish_data_loader=DataLoader(danish,None,danish_tags,None,max_len,model_danish)
        dutch_data_loader=DataLoader(dutch,None,dutch_tags,None,max_len,model_dutch)
        
        metaLearn=MetaLearn(hindi_data_loader,marathi_data_loader,sanskrit_data_loader,bhojpuri_data_loader,
                            magahi_data_loader,english_data_loader,german_data_loader,
                            dutch_data_loader,danish_data_loader,lossFunction,hidden_size,
                            epochs,inner_epoch,max_len,n_tokens,tokens_dict,dict_token,char_dict,n_chars,N,K,learning_rate)
        
        if args.resume_training:
                model=torch.load(args.checkpoint_path)
                metaLearn.epochs=model['epoch']
                metaLearn.load_state_dict(model['model'])
                
                if args.resume_training_type=='MAML':
                        metaLearn.train()
                        _=metaLearn.test()
                elif args.resume_training_type=='Reptile':
                        metaLearn.train_Reptile()
                        _=metaLearn.test()
                        
        elif args.load_model:
                metaLearn.load_state_dict(torch.load(args.model_path))
                _=metaLearn.test()           
        
        if training_mode=='MAML':
                metaLearn.train_MAML()
                _=metaLearn.test()
        elif training_mode=='Reptile':
                metaLearn.train_Reptile(epsilon)
                _=metaLearn.test()
        else:
                raise(NotImplementedError('This algorithm has not been implemented'))
                
def setup():
        parser=argparse.ArgumentParser('Metalearning argument parser')
        
        parser.add_argument('--learning_rate',type=float,default=0.01)
        parser.add_argument('--hidden_size',type=int,default=256)
        parser.add_argument('--N_way_learning',type=int,default=2)
        parser.add_argument('--training_mode',type=str,default='MAML')
        parser.add_argument('--epsilon',type=float,default=0.1)
        parser.add_argument('--epochs',type=int,default=1200)
        parser.add_argument('--load_model',type=bool,default=False)
        parser.add_argument('--model_path',type=str)
        parser.add_argument('--checkpoint_path',type=str)
        parser.add_argument('--resume_training_type',type=str,default='MAML')
        parser.add_argument('--resume_training',type=bool,default=False)
        parser.add_argument('--K_shot_learning',type=int,default=5)
        parser.add_argument('--inner_gradient_update',type=int,default=5)
        args=parser.parse_args()
        
        return args
    
if __name__=='__main__':
        args=setup()
        main(args)