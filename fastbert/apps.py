# from django.apps import AppConfig
#
#
# class FastbertConfig(AppConfig):
#     name = 'fastbert'


from django.apps import AppConfig
import html
import pathlib
import os
from deploy import settings
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout, GRU, BatchNormalization
from keras.callbacks import ModelCheckpoint
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l1
import json

# from fast_bert.prediction import BertClassificationPredictor

class WebappConfig(AppConfig):
    name = 'fastbert'
    wordnet_lemmatizer = WordNetLemmatizer()
    max_length= 17
    APP_DIR = os.path.join(settings.BASE_DIR,"fastbert")
    MODEL_DIR = os.path.join(APP_DIR,"model")
    BERT_PRETRAINED_PATH = MODEL_DIR+"/newModel0507.h5"
    DATA_FILE = MODEL_DIR+"/Dataset.csv"
    model = load_model(BERT_PRETRAINED_PATH)
    df = pd.read_csv(DATA_FILE, encoding = "latin1", names = ["Sentence", "Intent"])
    intent = df["Intent"]
    unique_intent = list(set(intent))
    unique_intent = ['faq.borrow_limit', 'Balance', 'commonQ.bot', 'faq.approval_time',
 'commonQ.query', 'faq.banking_option_missing', 'faq.apply_register',
 'faq.address_proof', 'commonQ.just_details', 'faq.biz_new', 'commonQ.how',
 'commonQ.assist', 'commonQ.not_giving', 'faq.application_process',
 'faq.aadhaar_missing', 'faq.biz_category_missing', 'faq.borrow_use',
 'faq.biz_simpler', 'commonQ.wait', 'contact.contact', 'faq.bad_service',
 'commonQ.name', 'MakeTransaction']
    sentences = list(df["Sentence"])
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        words.append([WordNetLemmatizer().lemmatize(i.lower()) for i in w])
    cleaned_words = words
    filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    token = Tokenizer(filters = filters)


    # def __init__(self):
    #     intent, unique_intent, sentences = load_dataset(DATA_FILE)
    #
    #
    #     self.wordnet_lemmatizer = WordNetLemmatizer()
    #
    #     APP_DIR = os.path.join(settings.BASE_DIR,"fastbert")
    #     MODEL_DIR = os.path.join(APP_DIR,"model")
    #     BERT_PRETRAINED_PATH = MODEL_DIR+"/newModel0507.h5"
    #     DATA_FILE = MODEL_DIR+"/Dataset.csv"
    #     NEW_MODEL_NAME = MODEL_DIR+'newTrainedModel.h5'
    #     # LABEL_PATH = Path("label/")
    #     # predictor = load_model(BERT_PRETRAINED_PATH)
    #     # intent, unique_intent, sentences = load_dataset(DATA_FILE)
    #     # print(unique_intent)
    #     cleaned_words = cleaning(sentences)
    #     word_tokenizer = create_tokenizer(cleaned_words)
    #     vocab_size = len(word_tokenizer.word_index) + 1
    #     max_length = max_lengthFunc(cleaned_words)
    #     encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
    #     padded_doc = padding_doc(encoded_doc, max_length)
    #     output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
    #     output_tokenizer.word_index
    #
    #
    #
    #     encoded_output = encoding_doc(output_tokenizer, intent)
    #     encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
    #     encoded_output.shape
    #     output_one_hot = one_hot(encoded_output)
    #     output_one_hot.shape
    #
    #     train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot,shuffle = True,test_size = 0.2)
    #
    #     print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
    #     print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))
    #
    #
    #     model = create_model(vocab_size, max_length)
    #     model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    #     model.summary()
    #
    #
    #     filename = model_name
    #     checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #
    #
    #     hist = model.fit(train_X, train_Y, epochs = 100,  validation_data = (val_X, val_Y),verbose=2)
    #
    #     scores = model.evaluate(train_X, train_Y, verbose=0)
    #     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #
    #     model.save(NEW_MODEL_NAME)


    @staticmethod
    def load_dataset(self, filename):
        df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
        intent = df["Intent"]
        unique_intent = list(set(intent))
        sentences = list(df["Sentence"])
        return (intent, unique_intent, sentences)

    @classmethod
    def predictions(self, text):
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
        test_word = word_tokenize(clean)
        test_word = [self.wordnet_lemmatizer.lemmatize(w.lower()) for w in test_word]
        word_tokenizer = self.create_tokenizer(self.cleaned_words,self.filters)
        test_ls = word_tokenizer.texts_to_sequences(test_word)

        #Check for unknown words
        if [] in test_ls:
            test_ls = list(filter(None, test_ls))
        test_ls = np.array(test_ls).reshape(1, len(test_ls))
        x = self.padding_doc(test_ls, self.max_length)
        print(x)
        pred = self.model.predict_proba(x)
        print(pred)
        final_op = self.get_final_output(pred,self.unique_intent)
        return final_op

    @classmethod
    def get_final_output(self, pred, classes):
        output = ''
        predictions = pred[0]
        print("PRedictions = {}".format(predictions))
        classes = np.array(classes)
        print(classes)
        ids = np.argsort(-predictions)
        print("ids with no.argsort(-predictions) = {}".format(ids))
        classes = classes[ids]
        print("classes with classes[ids] = {}".format(classes))
        predictions = -np.sort(-predictions)
        print(predictions)
        for i in range(pred.shape[1]):
            if predictions[i] > 0.50:
                output = "{} has confidence = {}".format(classes[i], (predictions[i]))
                data = {}
                data['intent'] = classes[i]
        return data


    @classmethod
    def cleaning(self, sentences):
        words = []
        for s in sentences:
            clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
            w = word_tokenize(clean)
            #lemmatizing
            words.append([self.wordnet_lemmatizer.lemmatize(i.lower()) for i in w])

        return words

    #creating tokenizer
    @classmethod
    def create_tokenizer(self, words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
        token = Tokenizer(filters = filters)
        token.fit_on_texts(words)
        return token

    #encoding list of words
    @classmethod
    def encoding_doc(self, token, words):
        return(token.texts_to_sequences(words))

    @classmethod
    def padding_doc(self, encoded_doc, max_length):
        return(pad_sequences(encoded_doc, maxlen = max_length, padding =   "post"))
    @classmethod
    def one_hot(self, encode):
        o = OneHotEncoder(sparse = False)
        return(o.fit_transform(encode))

    @classmethod
    def create_model(self, vocab_size, max_length):
        model = Sequential()

        model.add(Embedding(self.vocab_size, 128,
                input_length = self.max_length,  trainable = True))
        model.add(Bidirectional(LSTM(64, activity_regularizer=l1(0.001),dropout=0.5) ))
        model.add(Dense(64, activation = "relu"))

        model.add(Dense(64, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(23, activation = "softmax"))

        return model

    @classmethod
    def max_lengthFunc(words):
        return(len(max(words, key = len)))
