
#Importing NN modules
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Activation,Flatten,concatenate
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout,SpatialDropout1D,GlobalAveragePooling1D,GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import text,sequence
from keras.layers.wrappers import TimeDistributed

#Importing All other necessary modules
import pandas as pd
import seaborn as sns
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


#reading csv file
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.head()

#removing NA values
train_text=train["comment_text"].fillna("NA").values
test_text=test["comment_text"].fillna("NA").values
train=train.sample(frac=1)


y_train= train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values


#Parameters for Text processing
max_feature=50000
maxlen=100
embed_size=300

#text Processing
tokenizer=text.Tokenizer(num_words=max_feature)
tokenizer.fit_on_texts(list(train_text)+list(test_text))

X_train = tokenizer.texts_to_sequences(train_text)
X_test = tokenizer.texts_to_sequences(test_text)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


#Embedding File for training (You can download this from internet)
Embedding_File='crawl-300d-2M.vec'

def get_coefs(word,*arr):
    return word, np.asarray(arr,dtype='float32')

embeddings_index = {}

with open (Embedding_File,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs        


word_index = tokenizer.word_index
nb_words = min(max_feature, len(word_index)+1)
embedding_matrix = np.zeros((nb_words, embed_size))


for word, i in word_index.items():
    if i >= max_feature: 
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#define a cross validation funcyion
Callback=object
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

# Define neural network model 
def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_feature, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model=get_model()


#Train and test split
x_train, x_val, y_train, y_val=train_test_split(x_train,y,train_size=0.9,random_state=1)


RocAuc=RocAucEvaluation(validation_data=(x_val,y_val), interval=1)

#Model fit 
hist= model.fit(np.array(x_train),np.array(y_train),batch_size=32,epochs=2,validation_data=(x_val, y_val),verbose=2)


#Predictions
y_pred=model.predict(x_test,batch_size=1024)

