
# coding: utf-8

# In[21]:


import docx2txt
from glob import glob
from os import path
import random
# TODO: import packages 
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

def text_from_docx(fn):
    text = docx2txt.process(fn)
    text = re.findall('[a-zA-Z0-9]+',text)
    return ' '.join(text)


def load_data():
    all_pos_txt = []
    for fn in glob(path.join('DATA', '01', '*.docx')):
        text = text_from_docx(fn)
        all_pos_txt.append(text)
    all_neg_txt = []
    for fn in glob(path.join('DATA', '00', '*.docx')):
        text = text_from_docx(fn)
        all_neg_txt.append(text)

    return all_pos_txt, all_neg_txt

def process(all_pos_txt, all_neg_txt):
    all_pos_df = pd.DataFrame(all_pos_txt,columns=['text'])
    all_pos_df['label'] = 1
    all_neg_df = pd.DataFrame(all_neg_txt,columns=['text'])
    all_neg_df['label'] = 0
    train_test_df = pd.concat([all_pos_df,all_neg_df])
    text_list = list(train_test_df['text']) 
    tfidf = TfidfVectorizer(max_features=100)
    tf_idf = tfidf.fit_transform(text_list).toarray()
    for i in range(tf_idf.shape[1]):
        train_test_df['tfidf%d' % i] = tf_idf[:,i] 
    return train_test_df

def train_and_eval():
    all_pos, all_neg = load_data()
    train_test_df    = process(all_pos, all_neg)
    X = train_test_df.iloc[:,2:].values
    y = train_test_df['label'].values
    
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.1,stratify=y)
    
    # Classifier
    clf = RandomForestClassifier(100) #  
    model = clf.fit(X_train, y_train)
    y_val_out = model.predict(X_val)
    y_train_out = model.predict(X_train)
    # calculate accuracy
    acc_val = np.mean(y_val_out == y_val)  # calculate accuracy
    acc_train = np.mean(y_train_out == y_train)
    print('Train acc = {}, Val acc = {}'.format(acc_train, acc_val))
    print('Saving model...')
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
if __name__ == '__main__':
    train_and_eval()

