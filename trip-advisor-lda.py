# Author: Harshit Sharma <hsharma1205@gmail.com>
# License: MIT
# Python 2.7

'''
This script takes a single JSON file from the Trip Advisor reviews dataset.
(Link to dataset: http://times.cs.uiuc.edu/~wang296/Data). Refer to the 
ipython notebook in the repo to understand the workflow.

Make sure the "data_file" variable is updated to the file chosen for review after
downloading the dataset.

The following process is followed:

 * Dataframe is created using pandas. All the review text is added to a list.
 * Text pre-processing is done. Tokenisation, removal of stopwords is performed.
 * A list for cleaned reviews is created.
 * Creating a dictionary and corpus that can be used to train an LDA model.
 * LDA is trained.
 * pyLDAvis is used to visualise the topics.

'''

import gensim
import pandas as pd
import nltk
import re
from nltk.tokenize import RegexpTokenizer
import pyLDAvis.gensim as gensimvis
from gensim import corpora, models, similarities

#-------------------------------------------------------
# Reading the JSON file and creating a list of reviews.
#-------------------------------------------------------

# Path to the json file in the TripAdvisor Dataset
data_file = "/PATH/TO/THE/JSON/File"  

# Reading a json file using pandas
pd.read_json("data_file", typ = "series")

with open(data_file, "rb") as f:
    data = f.readlines()

# Reading the data to a dataframe

data_json_str = "["+','.join(data) + "]"
data_df = pd.read_json(data_json_str)

num_reviews_tpadv = len(data_df["Reviews"][0])

all_reviews = []

# Adding all the reviews to all_reviews list 

for i in range(num_reviews_tpadv):
    all_reviews.append(data_df["Reviews"][0][i]["Content"])

#-----------------------------------------------------------
# Text Preprocessing and creating a list of cleaned reviews
#-----------------------------------------------------------

# Text Preprocessing to clean the data. Removing every character except for alphabets, 
# tokenising and removing stopwords.

tokenizer = RegexpTokenizer(r'\w+')

def clean_review(text):
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    words = tokenizer.tokenize(letters_only.lower())
    stops = set(nltk.corpus.stopwords.words("english")) 
    _words = [w for w in words if not w in stops]  
    return _words

clean_reviews = []

# Adding cleaned reviews to clean_reviews list

for i in range(num_reviews_tpadv):
    clean_reviews.append(clean_review(data_df["Reviews"][0][i]["Content"]))

#----------------------------------------------------
# Creating Dictionary and Corpus to train LDA model
#----------------------------------------------------

dictionary = corpora.Dictionary(clean_reviews)
dictionary.compactify()

# convert tokenized documents to vectors

corpus = [dictionary.doc2bow(doc) for doc in clean_reviews]


# Training lda using number of topics set = 10 (which can be changed)

lda = models.LdaModel(corpus, id2word = dictionary,
                        num_topics = 10,
                        passes = 20,
                        alpha = "auto")

#--------------------------------------------
# Visualising topic models using pyLDAvis
#--------------------------------------------

vis_data = gensimvis.prepare(lda, corpus, dictionary)

pyLDAvis.display(vis_data)




