from  data_loader import load_sentences,get_sentences,get_tokens

max_len=116

marathi_train,marathi_test,hindi_train,hindi_test=load_sentences()
tokens_dict,dict_token,n_tokens=get_tokens(marathi_train)
marathi_train,marathi_test,marathi_train_tags,marathi_test_tags=get_sentences(marathi_train,marathi_test,tokens_dict,max_len)

s=set()

for sentence in marathi_train:
    for word in sentence:
            for character in word:
                    s.add(character)
                    
print(s)
  
k=set()
t=set()               
for sentence in marathi_test:
    for word in sentence:
            for character in word:
                   if character not in s:
                           print(sentence)
                           k.add(character)
                           
print(k)