def preprocess(data,words,max_len):
        xtrain=[]
        ytrain=[]
        for sentence in data:
                a=[]
                b=[]
                for token in sentence:
                        if token.form is not None:
                                a.append(words.word_to_int[token.form])
                                b.append(words.token_to_int[token.upos])
                for _ in range(max_len-len(a)):
                        a.append(words.word_to_int['PAD'])
                        b.append(words.token_to_int['X'])
                xtrain.append(a)
                ytrain.append(b)  
        return xtrain,ytrain
    
def preprocess_2(sentences):
        for sentence in sentences:
                for token in sentence:
                        if token.upos is None:
                                token.upos='X'
        return sentences