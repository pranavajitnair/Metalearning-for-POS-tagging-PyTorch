from data_loader import load_sentences,get_sentences,get_tokens

marathi_train,marathi_test,hindi_train,hindi_test=load_sentences()

tokens_dict,dict_token,n_tokens=get_tokens(marathi_train)

marathi_train,marathi_test,marathi_train_tags,marathi_test_tags=get_sentences(marathi_train,marathi_test,tokens_dict,116)

import tensorflow as tf
import tensorflow_hub as hub

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

embeddings = embed(
    ["My name is pranav"],
    signature="default",
    as_dict=True)["elmo"]


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  x = sess.run(embeddings)


import sys
tf.print(embeddings, output_stream=sys.stderr)