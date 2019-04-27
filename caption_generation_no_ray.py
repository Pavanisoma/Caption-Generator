from utils import load_flick8r_descriptions, load_partly_descriptions
from Inception_model import load_image_files, load_features_from_file, extract_features_inceptionv3
from sequence_utils import create_tokenizer, max_sequence_length, vocabulary_size, load_glove_file, create_embedding_matrix
from model_no_ray import pad_sequences, define_model, data_generator
import h5py
import numpy as np
from numpy import argmax
from keras.models import load_model


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text[9:]


def main():
    # load all data
    description_dict = load_flick8r_descriptions()
    train_files, test_files = load_image_files()

    # # divide descriptions pool for train and test data
    train_description_dict = load_partly_descriptions(description_dict, train_files)
    test_description_dict = load_partly_descriptions(description_dict, test_files)
    
    # #Extract features 
    #extract_features_inceptionv3(train_files, "encoded_train_images_inceptionv3.p")
    #extract_features_inceptionv3(test_files, "encoded_test_images_inceptionv3.p")

    # # load image features which are already extracted (saved in file)
    train_features_dict = load_features_from_file('encoded_train_images_inceptionv3.p')
    test_features_dict = load_features_from_file('encoded_test_images_inceptionv3.p')

    # # prepare tokenizer, vocabulary for word embedding
    tokenizer = create_tokenizer(train_description_dict)
    max_length = max_sequence_length(description_dict)
    vocab_size = vocabulary_size(tokenizer)
    embeddings_index = load_glove_file('glove/glove.6B.300d.txt')
    embedding_matrix = create_embedding_matrix(tokenizer, embeddings_index)

    model = define_model(vocab_size, max_length, embedding_matrix)

    epochs = 20
    steps = len(train_description_dict)
    for i in range(epochs):
        generator = data_generator(train_description_dict, train_features_dict, tokenizer)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save('model_' + str(i) + '.h5')
        model.save_weights('weights_'+ str(i)+'.h5',overwrite=True)

    print(test_features_dict['3385593926_d3e9c21170'])
    caption = generate_desc(model, tokenizer, test_features_dict['3385593926_d3e9c21170'], max_length)
    print(caption)

if __name__ == '__main__':
    main()