#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import nltk
import numpy as np


# In[ ]:


#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS 
from PIL import Image
import matplotlib_venn as venn
#nlp
import string
import re
from nltk.corpus import stopwords
from scipy.sparse import hstack
#featureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score


# In[ ]:


#settings
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#reading csv file
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.tail(10)


# In[ ]:


#data introspection
print(len(test))


# In[ ]:


train.info()


# In[ ]:


x=train.iloc[:,2:7].sum()


# In[ ]:


x


# In[ ]:


rowsums=train.iloc[:,2:7].sum(axis=1)


# In[ ]:


train['clean']=(rowsums==0)
train['clean'].sum()
print("Total comments = ",len(train))
print("Total clean comments = ",train['clean'].sum())
print("Total tags =",x.sum())


# In[ ]:


print("Check for missing values in comment dataset")
null_check=train.isnull().sum()
print(null_check)
print("filling NA with \"unknown\"")
train["comment_text"].fillna("unknown", inplace=True)


# In[ ]:


x=train.iloc[:,2:].sum()
#plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('Frequncy of occurance', fontsize=12)
plt.xlabel('Type ', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[ ]:


x=rowsums.value_counts()

#plot
plt.figure(figsize=(8,4))
ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[2])
plt.title("Multiple tags per comment")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of tags ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[ ]:


temp_df=train.iloc[:,2:-1]
# filter temp by removing clean comments
# temp_df=temp_df[~train.clean]

corr=temp_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True)


# In[ ]:


#Crosstab
# Since technically a crosstab between all 6 classes is impossible to vizualize, lets take a 
# look at toxic with other tags
main_col="toxic"
corr_mats=[]
for other_col in temp_df.columns[1:]:
    confusion_matrix = pd.crosstab(temp_df[main_col], temp_df[other_col])
    corr_mats.append(confusion_matrix)
out = pd.concat(corr_mats,axis=1,keys=temp_df.columns[1:])

out


# In[ ]:


#Checking for Toxic and Severe toxic for now
col1="toxic"
col2="severe_toxic"
confusion_matrix = pd.crosstab(temp_df[col1], temp_df[col2])
print("Confusion matrix between toxic and severe toxic:")
print(confusion_matrix)
#new_corr=cramers_corrected_stat()
#print("The correlation between Toxic and Severe toxic using Cramer's stat=",new_corr)


# In[ ]:


print("toxic:")
print(train[train.severe_toxic==1].iloc[3,1])


# In[ ]:


print("severe_toxic:")
print(train[train.severe_toxic==1].iloc[8,1])


# In[ ]:


print("Threat:")
print(train[train.threat==1].iloc[1,1])


# In[ ]:


print("Obscene:")
print(train[train.obscene==1].iloc[2,1])


# In[ ]:


print("identity_hate:")
print(train[train.identity_hate==1].iloc[4,1])


# In[ ]:


subset=train[train.toxic==1]
text=subset.comment_text.values

stopword=set(STOPWORDS)


# In[ ]:


plt.figure(figsize=(20,10))
wordcloud=WordCloud(background_color='WHITE',mode="RGB",width=500,height=400,max_words=1000,stopwords=stopword)
wordcloud.generate("".join(text))
plt.imshow(wordcloud.recolor(colormap='Blues',random_state=240))
plt.axis("off")
plt.title("Frequent words in Toxic comments")


# In[ ]:


#Severely toxic comments
plt.subplot(222)
plt.figure(figsize=(20,10))
subset=train[train.severe_toxic==1]
text2=subset.comment_text.values
wc= WordCloud(background_color="White",max_words=1000,stopwords=stopword,width=500,height=400)
wc.generate(" ".join(text2))
plt.axis("off")
plt.title("Frequent words in Severe Toxic Comments", fontsize=16)
plt.imshow(wc.recolor(colormap= 'Reds' , random_state=244))


# In[ ]:


#Obscence comments comments
plt.subplot(222)
plt.figure(figsize=(20,10))
#severe_toxic_mask=np.array(Image.open("severe toxic.jpg"))
#severe_toxic_mask=severe_toxic_mask[:,:,1]
subset=train[train.obscene==1]
text2=subset.comment_text.values
wc= WordCloud(background_color="White",max_words=1000,stopwords=stopword,width=500,height=400)
wc.generate(" ".join(text2))
plt.axis("off")
plt.title("Words frequented in Obscence", fontsize=16)
plt.imshow(wc.recolor(colormap= 'Reds' , random_state=244))


# In[ ]:


#Identity hate
plt.subplot(222)
plt.figure(figsize=(20,10))
#severe_toxic_mask=np.array(Image.open("severe toxic.jpg"))
#severe_toxic_mask=severe_toxic_mask[:,:,1]
subset=train[train.identity_hate==1]
text2=subset.comment_text.values
wc= WordCloud(background_color="White",max_words=1000,stopwords=stopword,width=500,height=400)
wc.generate(" ".join(text2))
plt.axis("off")
plt.title("Frequent word in identity hate comments", fontsize=16)
plt.imshow(wc.recolor(colormap= 'Reds' , random_state=244))


# In[ ]:


plt.subplot(222)
plt.figure(figsize=(20,10))
#severe_toxic_mask=np.array(Image.open("severe toxic.jpg"))
#severe_toxic_mask=severe_toxic_mask[:,:,1]
subset=train[train.threat==1]
text2=subset.comment_text.values
wc= WordCloud(background_color="White",max_words=1000,stopwords=stopword,width=500,height=400)
wc.generate(" ".join(text2))
plt.axis("off")
plt.title("Frequent words in threat comment", fontsize=16)
plt.imshow(wc.recolor(colormap= 'RdYlBu' , random_state=244))


# In[ ]:


plt.subplot(222)
plt.figure(figsize=(20,10))
#severe_toxic_mask=np.array(Image.open("severe toxic.jpg"))
#severe_toxic_mask=severe_toxic_mask[:,:,1]
subset=train[train.insult==1]
text2=subset.comment_text.values
wc= WordCloud(background_color="White",max_words=1000,stopwords=stopword,width=500,height=400)
wc.generate(" ".join(text2))
plt.axis("off")
plt.title("Words frequented in insult comments", fontsize=16)
plt.imshow(wc.recolor(colormap= 'Greens' , random_state=244))


# **Text Processing**
# 

# In[ ]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[ ]:


train['comment_text'].apply(text_process)


#  ### Logestic regression for classification model

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


train_text=train['comment_text']
test_text=test['comment_text']
all_text=pd.concat([train_text,test_text],axis=0)


# In[ ]:


word_vectorizer= TfidfVectorizer(
sublinear_tf=True,
strip_accents='unicode',
stop_words='english',
ngram_range=(1,1),
max_features=10000
)


# In[ ]:


word_vectorizer.fit(all_text)


# In[ ]:


train_word_features=word_vectorizer.transform(train_text)
test_word_features=word_vectorizer.transform(test_text)


# In[ ]:


char_vectorizer=TfidfVectorizer(
analyzer='char',
sublinear_tf=True,
strip_accents='unicode',
stop_words='english',
ngram_range=(2,6),
max_features=50000
)


# In[ ]:


char_vectorizer.fit(all_text)


# In[ ]:


train_char_features=char_vectorizer.transform(train_text)
test_char_features=char_vectorizer.transform(test_text)


# In[ ]:


from scipy.sparse import hstack
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])


# In[ ]:


scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})


# In[ ]:


optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']


# In[ ]:


for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]


# In[ ]:


print('Total CV score is {}'.format(np.mean(scores)))


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




