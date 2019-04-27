from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk import word_tokenize, translate
from numpy import array
from numpy import asarray
from numpy import zeros
import pickle
from keras.models import load_model


# convert a dictionary of clean descriptions to a list of descriptions
def get_all_desc(description_dict):
    all_desc = list()
    for key in description_dict.keys():
        [all_desc.append(d) for d in description_dict[key]]
    return all_desc
 
# fit a tokenizer given caption descriptions
# get the word counts of the vocabulary
def create_tokenizer(description_dict):
    lines = get_all_desc(description_dict)
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the max length of descriptions
def max_sequence_length(description_dict):
    lines = get_all_desc(description_dict)
    return max(len(sequence.split()) for sequence in lines)

def vocabulary_size(tokenizer):
    return len(tokenizer.word_index) + 1

def load_glove_file(file='glove/glove.6B.300d.txt'):
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index
    

def create_embedding_matrix(tokenizer, embeddings_index):
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocabulary_size(tokenizer), 300))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Embedding vector created')
    print(embedding_matrix.shape)
    return embedding_matrix


# def testing(caption,weight, img_dir, beam_size=3):
#     captions = {}
#     with open(img_dir, 'r') as images_path:
#         images = images_path.read().strip().split('\n')
#     encoded_images = pickle.load(open("encoded_test_images_inceptionv3.p", "rb"))
#     #model = e.create_model(ret_model=True)
#     #model.load_weights(weight)

#     prediction = open('predicted_captions.txt', 'w')
#     for count, image in enumerate(images):
#         image = encoded_images[image]
#         captions[image] = generate_desc(model, tokenizer, test_features_dict['3385593926_d3e9c21170'].reshape((1,2048)), max_length)
#         prediction.write(image + "\t" + str(caption))
#         prediction.flush()
#     prediction.close()

#     captions_path = open('Flickr8k_text/Flickr8k.token.txt', 'r')
#     captions_text = captions_path.read().strip().split('\n')
#     cap_pair = {}
#     for i in captions_text:
#         i = i.split("\t")
#         i[0] = i[0][:len(i[0]) - 2]
#         try:
#             cap_pair[i[0]].append(i[1])
#         except:
#             cap_pair[i[0]] = [i[1]]
#     captions_path.close()

#     h = []
#     r = []
#     for image in images:
#         h.append(captions[image])
#         r.append(cap_pair[image])

#     return BLEU(h, r)

# def BLEU(h, r):
#     return translate.bleu_score.corpus_bleu(r, h)

def main():
    # sample data
    train_description_dict = { '2671602981_4edde92658': ['a little girl in a bathing suit leaps up in the water', 
    'a little girl wearing a black tankini is jumping in the air with water in the background', 
    'a young girl in a swimming suit jumps into a body of water', 
    'a young girl jumping out of the water', 
    'the girl in the bathing suit is poised in midair next to the blue water']}
    tokenizer = create_tokenizer(train_description_dict)
    print(tokenizer.texts_to_sequences(['a young girl']))

if __name__ == '__main__':
   main()