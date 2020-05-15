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
        
        epochs=args.epochs
        inner_epoch=args.N_shot_learning
        test_size_hindi=args.test_size_hindi
        test_size_marathi=args.test_size_marathi
        max_len=116
        
        marathi_train,marathi_test,hindi_train,hindi_test=load_sentences()
        tokens_dict,dict_token,n_tokens=get_tokens(marathi_train)
        
        marathi_train,marathi_test,marathi_train_tags,marathi_test_tags=get_sentences(marathi_train,marathi_test,tokens_dict,max_len)
        hindi_train,hindi_test,hindi_train_tags,hindi_test_tags=get_sentences(hindi_train,hindi_test,tokens_dict,max_len)
        
        char_dict,n_chars=get_characters(marathi_train+hindi_train)
        
        model_hindi=gs.Word2Vec(hindi_train+hindi_test,min_count=1,size=hidden_size)
        model_marathi=gs.Word2Vec(marathi_test+marathi_train,min_count=1,size=hidden_size)
        
        hindi_data_loader=DataLoader(hindi_train,hindi_test,hindi_train_tags,hindi_test_tags,max_len,model_hindi)
        marathi_data_loader=DataLoader(marathi_train,marathi_test,marathi_train_tags,marathi_test_tags,max_len,model_marathi)
        
        metaLearn=MetaLearn(hindi_data_loader,marathi_data_loader,lossFunction,hidden_size,epochs,inner_epoch,max_len,n_tokens,tokens_dict,dict_token,char_dict,n_chars)
        
        if training_mode=='MAML':
                metaLearn.train_MAML()
        elif training_mode=='Reptile':
                metaLearn.train_Reptile(epsilon)
        elif training_mode=='FOMAML':
                metaLearn.train_FOMAML()
        else:
                raise(NotImplementedError('This algorithm has not been implemented'))
                
        metaLearn.test(test_size_marathi,'marathi')
        metaLearn.test(test_size_hindi,'hindi')
        
def setup():
        parser=argparse.ArgumentParser('Metalearning argument parser')
        
        parser.add_argument('--learning_rate',type=float,default=0.01)
        parser.add_argument('--hidden_size',type=int,default=256)
        parser.add_argument('--test_size_marathi',type=int,default=47)
        parser.add_argument('--test_size_hindi',type=int,default=1600)
        parser.add_argument('--N_shot_learning',type=int,default=5)
        parser.add_argument('--training_mode',type=str,default='MAML')
        parser.add_argument('--epsilon',type=float,default=1.0)
        parser.add_argument('--epochs',type=int,default=1200)
        
        args=parser.parse_args()
        
        return args
    
if __name__=='__main__':
        args=setup()
        main(args)