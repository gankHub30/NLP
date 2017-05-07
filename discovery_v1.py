import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
#import mpld3
from nltk.stem.snowball import SnowballStemmer
from __future__ import print_function
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

#Stopwords
with open('/Users/tganka/Desktop/twitter/stopWords_v2.txt', 'r') as text:
	stopWords = []
	for line in text:
		for word in line.split():
			stopWords.append(word.decode('utf-8'))
#Stemmer
stemmer = SnowballStemmer("english")

'''Clean up data for Analysis'''
#Load Random Set of 5000 Chats	
print("Loading dataset...")
dataset = pd.read_csv("/Users/tganka/Documents/Analytics/TMO/tmo_trans_analysis_20160602_20160608.csv", header=0, delimiter="|", quoting=2, encoding='utf-8' )
tweets = dataset["tweet"]

#additional cleaning
def text_to_words( raw_text ):
    # Function to convert a raw transcript to a string of words
    # The input is a single string (a raw transcript), and 
    # the output is a single string (a preprocessed transcript)
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
	letters_only = re.sub("[^a-zA-Z]", " ", raw_text) 
	letters_only_re = letters_only.replace("do nt","dont")
    #
    # 3. Convert to lower case, split into individual words
	words = nltk.word_tokenize(letters_only_re.lower())                           
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
	stops = set(stopWords) #stopWords
    # 
    # 5. Remove stop words
	meaningful_words = [word for word in words if word not in stops]
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
	return( " ".join( meaningful_words ))  

clean_text=[]
num_transcripts = len(dataset["id"])
print ("Cleaning and parsing the twitter data...\n")
for i in xrange(0,num_transcripts):
    if( (i+1) % 1000 == 0 ):
        print ("Transcript %d of %d\n" % (i+1, num_transcripts))
    clean = text_to_words( tweets[i] )
    clean_text.append( clean )

#Here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(tweets):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(tweets) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(tweets):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(tweets) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in clean_text:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'chats', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

'''Prepare Model'''

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.9999, max_features=200000,
                                 min_df=2, use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(2,2))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(clean_text) #fit the vectorizer to chats data

print(tfidf_matrix.shape)

#terms
terms = tfidf_vectorizer.get_feature_names()

#Top Terms
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(tokenizer=tokenize_only, ngram_range=(3, 3))

# Don't need both X and transformer; they should be identical
X = vectorizer.fit_transform(clean_text)
word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
word_freq_df['frequency'] = word_freq_df['occurrences']/np.sum(word_freq_df['occurrences'])
print(word_freq_df.sort('occurrences',ascending = False).head(n=20))


#Cosine dist

dist = 1 - cosine_similarity(tfidf_matrix)

#KMeans for Initial Discovery
num_clusters = 8

km = KMeans(n_clusters=num_clusters)

%time km.fit(tfidf_matrix)

#Save model / Load model
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()


#create dictionary of classification, cluster, 

tweets = {'tweets': clean_text, 'cluster': clusters}
frame = pd.DataFrame(tweets, index = [clusters] , columns = ['tweets', 'cluster'])

frame['cluster'].value_counts().sort_index() #number of chats per cluster (clusters from 0 to 7)
'''
0     671
1     212
2     166
3     771
4     930
5    6347
6     282
7     848
'''

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
print()
print()

'''
Cluster 0 words: dm, send, response, follow, filled, guys

Cluster 1 words: data, plan, unlimited, slow, roaming, worked

Cluster 2 words: phone, worked, days, paying, number, service

Cluster 3 words: helping, dm, phone, guys, mobile, issue

Cluster 4 words: ny, atrocious, usa, markhamade, applesupport, replace

Cluster 5 words: service, customer, worked, phone, issue, good

Cluster 6 words: https, dm, guys, message, going, time

Cluster 7 words: mobile, loveing, htc, customer, speeds, dm
'''

#Multidimensional scaling

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()

#set up cluster names using a dict
cluster_names = {0: 'Account Login', 
                 1: 'Cancel Service', 
                 2: 'TV/Internet Bundle', 
                 3: 'Account Number', 
                 4: 'Return Modem',
                 5: 'Internet Speed'
                 6: 'Phone Service'
                 7: 'Payment'}

#Hierarchical Clustering (Ward)
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=terms);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

#Gaussian Mixture Model
from sklearn.mixture import GMM

gmm_matrix = GMM(dist) #define the linkage_matrix using ward clustering pre-computed distances






