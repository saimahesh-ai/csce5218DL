#!/usr/bin/env python
# coding: utf-8

# # csce5222 feature engg assignment1

# IMPORT functions

# In[1]:


import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
import requests


# In[ ]:





# article link

# In[2]:


link = "https://www.bbc.com/news/technology-56906145"


# In[3]:


page = requests.get(link)


# import beautifulsoup

# In[4]:


from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content,'html.parser')


# In[5]:


soup


# In[6]:


contents  = soup.find_all('p')


# In[7]:


len(contents)


# In[ ]:





# WEB SCRAPING process

# In[10]:



news_contents = []
list_paragraphs = []

for p in np.arange(0,len(contents)):
    paragraph = contents[p].get_text()
    list_paragraphs.append(paragraph)
    final_article = "".join(list_paragraphs)
news_contents.append(final_article)


# In[11]:


#final_article[0]
News_contents= news_contents[0]
News_contents


# In[12]:


News_contents=(((((News_contents.replace("\r", " ")).replace("\n", " ")).replace("    ", " ")).replace('"', '')).lower())
News_contents.replace("'s", "")
df = pd.DataFrame({News_contents})
df['Content'] = News_contents
df['Content_Parsed_1'] =(((News_contents.replace("\r", " ")).replace("\n", " ")).replace("    ", " "))
df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()


# In[13]:


punctuation_signs = list("?:!.,;")
df['Content_Parsed_3'] = df['Content_Parsed_2']

for punct_sign in punctuation_signs:
    df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')


# In[14]:


df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")


# STEMMING AND LEMMATIZZATION

# In[15]:


# Downloading punkt and wordnet from NLTK
nltk.download('punkt')
print("------------------------------------------------------------")
nltk.download('wordnet')


# In[16]:


wordnet_lemmatizer = WordNetLemmatizer()


# In[17]:


nrows = len(df)
lemmatized_text_list = []

for row in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text = df.loc[row]['Content_Parsed_4']
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)


# In[18]:


df['Content_Parsed_5'] = lemmatized_text_list


# STOP WORDS

# In[19]:


# Downloading the stop words list
nltk.download('stopwords')


# In[20]:


# Loading the stop words in english
stop_words = list(stopwords.words('english'))


# In[21]:


stop_words[0:10]


# In[22]:


example = "me eating a meal"
word = "me"

# The regular expression is:
regex = r"\b" + word + r"\b"  # we need to build it like that to work properly

re.sub(regex, "StopWord", example)


# In[23]:


df['Content_Parsed_6'] = df['Content_Parsed_5']

for stop_word in stop_words:

    regex_stopword = r"\b" + stop_word + r"\b"
    df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')


# In[24]:


df.head(1)


# In[25]:


list_columns = ["Content_Parsed_6"]
df = df[list_columns]

df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})


# In[26]:


X_new= df['Content_Parsed']


# In[27]:



loaded_model= pickle.load(open('C:\\Users\\Mahesh\\Documents\\Latest-News-Classifier-master\\0. Latest News Classifier\\04. Model Training\\Models\\best_mnbc.pickle','rb'))
X_train =pickle.load(open('C:\\Users\\Mahesh\\Documents\\Latest-News-Classifier-master\\0. Latest News Classifier\\03. Feature Engineering\\Pickles/X_train.pickle','rb'))
y_train =pickle.load(open('C:\\Users\\Mahesh\\Documents\\Latest-News-Classifier-master\\0. Latest News Classifier\\03. Feature Engineering\\Pickles/y_train.pickle','rb'))


# TEXT REPRESENTATION

# In[28]:


# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300


# In[29]:


tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_new = tfidf.transform(X_new).toarray()

print(features_new.shape)


# In[30]:


features_new


# In[31]:


loaded_model.predict(features_new)


# In[ ]:




