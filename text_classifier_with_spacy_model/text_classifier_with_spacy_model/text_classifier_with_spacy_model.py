
# Text classifier using the spacy pacage 
# The dataset is made up of two classes in a 4 to 1 ratio.
# In the following script I create a classifier using the textcat models of the spacy library as a basis.

import pandas as pd 
import numpy as np
import spacy, re, time 
from collections import Counter
from sklearn.utils import shuffle
from spacy.util import minibatch, compounding

nlp_cl = spacy.load('it_core_news_sm')


# Text preprocessing: delete email addresses, numbers, punctuation and spaces
def cleanText ( text ):            
    text = text.lower()
    text = text.replace('"', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')            
    text = re.sub( r'\S+@\S+', ' ', text)
    text =re.sub( '\w*\d\w*', ' ', text) 
    text = re.sub( r'[/ () {} \[\] ;: °_ \- <> º€\+\*\$#\| ]', ' ', text)
    text = text.replace('\\', ' ')
    text = re.sub( r' +', ' ', text )
    return text


# Create the labels in the spacy format: a dictionary in which I select the positive value for the current label and all other negatives
def makeLabels ( num ):
    dk = {}
    if num == 0:
        dk = { 'positive': True, 'negative': False}
    else:
        dk = {'positive':False, 'negative': True}
    return dk 


# split dataset in training and test set 
def makeSet ( df ):
    for i in range(0, 3):
        df = shuffle( df, random_state=33)
    x = list( df['corpus'])
    y = list( df['labels'])
    sp = int( len(y)*0.8)
    x_tr, x_ts = x[:sp], x[sp:]
    y_tr, y_ts = y[:sp], y[sp:]
    return  x_tr, y_tr, x_ts, y_ts 


# Create the training set in the spacy format: a list of tuples in which a tuple contains the text and the dictionary of the respective label
def dataBlok(x, y):
    blk = []
    for i in range(0, len(y)):
        text = x[i]
        label = y[i]
        dic = {'cats': label}
        w = (text, dic)
        blk.append(w)
    return blk 


# Evaluate the model: calculation of accuracy, precision, recall and f1
def performance(docs, textcat, texts, cats):    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, doc in enumerate( textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items(): 
            if label not in gold:
                continue 
            if label == 'NEGATIVE':
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp +=1
            elif  score >= 0.5 and gold[label] < 0.5:
                fp+= 1
            elif  score < 0.5 and gold[label] < 0.5:
                tn +=1
            elif score <  0.5 and gold[label] >= 0.5:
                fn += 1

    ac = (tp)/ len(y_ts)
    precision = tp/( tp+fp)
    recall = tp/( tp+fn)
    f1 = 0
    if precision + recall != 0: 
        f1 = 2*( precision* recall)/(precision + recall)
    return  ac, precision, recall, f1

print("go")
df = pd.read_excel('...\\corpus_tiket_11.xlsx' )
df = df[['Descrizione', 'targhet_class']]
print(" df size  ",df.shape)

df['corpus'] = df['Descrizione'].apply( cleanText )
print(" corpus ")
df['labels'] = df['targhet_class'].apply( makeLabels)
print(" spacy labels ",df['labels'][0])

x_tr, y_tr, x_ts, y_ts = makeSet( df)
print(" train size: ",len(y_tr)," test size: ",len(y_ts))
trainBlok =  dataBlok( x_tr, y_tr)
print(" train format \n",trainBlok[:3])

st = time.time()

# make model 
nlp = spacy.blank( 'it')
textcat = nlp.create_pipe( 'textcat', config= {"exclusive_classes": True, "architecture":"simple_cnn"})
nlp.add_pipe( textcat)
print(" spacy container ",nlp.pipe_names)
textcat.add_label( 'positive')
textcat.add_label('negative')

print("start training ")
optimizer = nlp.begin_training()
for i in range(0, 10):
    print(" epoch ",i)
    losses = {}
    batches = minibatch( trainBlok, size=64)
    z = 0
    for bc in batches:
        texts, labels = zip(* bc) 
        nlp.update( texts, labels, sgd=optimizer, drop=0.2, losses=losses)
        z+=1
        #if z%100 == 0: print(" bc ",z)
    print("batches are ",z)

et = time.time()
print(" the training time is %.2f seconds  " %(et-st))

# test the model 
docs = [nlp.tokenizer( text ) for text in x_ts]
textcat = nlp.get_pipe( 'textcat')

with  textcat.model.use_params( optimizer.averages ):
    accuracy, precision, recall, f1 = performance(docs,  textcat, x_ts, y_ts)
    print("test model: accuracy = %.4f, precision= %.4f, recall =%.4f, f1 =%.4f  " (%(accuracy,  precision, recall, f1)

    print("end ")
