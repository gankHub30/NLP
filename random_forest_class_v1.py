'''Random Forest'''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  
 
# Read the train data
train = pd.read_csv("D:/classification/rand_500.csv", header=0, delimiter="\t", \
                   quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print train.shape

# Create an empty list and append the clean reviews one by one
num_transcripts = len(train["transcript_id"])
clean_train_text = [] 

print "Cleaning and parsing the test set transcripts...\n"
for i in xrange(0,num_transcripts):
    if( (i+1) % 1000 == 0 ):
        print "Transcript %d of %d\n" % (i+1, num_transcripts)
    clean_text = text_to_words( train["text"][i] )
    clean_train_text.append( clean_text )

# Get a bag of words for the test set, and convert to a numpy array
train_data_features = vectorizer.transform(clean_train_text)
train_data_features = train_data_features.toarray()

print "Training the random forest..."

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["label"] )
#training results
result_train = forest.predict(train_data_features)
#training output
output_train = pd.DataFrame( data={"transcript_id":train["transcript_id"], "label":result_train} )


#Collect Test Data to Model
# Read the test data
test = pd.read_csv("D:/classification/2015-09-01_2015-11-30_rand_100.csv", header=0, delimiter="\t", \
                   quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_transcripts = len(test["transcript_id"])
clean_test_text = [] 

print "Cleaning and parsing the test set transcripts...\n"
for i in xrange(0,num_transcripts):
    if( (i+1) % 1000 == 0 ):
        print "Transcript %d of %d\n" % (i+1, num_transcripts)
    clean_text = text_to_words( test["text"][i] )
    clean_test_text.append( clean_text )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_text)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "transcript_id" column and
# a "label" column
output = pd.DataFrame( data={"transcript_id":test["transcript_id"], "label":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "D:/Classification/BoW_RF_model.csv", index=False, quoting=3 )

#Evaluate Model Performance
'''You typically plot a confusion matrix of your test set (recall and precision), and report an F1 score on them.
If you have your correct labels of your test set in y_test and your predicted labels in pred, then your F1 score is: '''
from sklearn import metrics
y_test = test["label"]
pred = output["label"]
y_train = train["label"]
pred_train = output_train["label"]
# testing score
score = metrics.f1_score(y_test, pred, pos_label=['sales','service']) #0.8306715063520872
# training score
score_train = metrics.f1_score(y_train, pred_train, pos_label=['sales','service']) #1.0
'''You can also use accuracy: '''
pscore = metrics.accuracy_score(y_test, pred) #0.84999999999999998
pscore_train = metrics.accuracy_score(y_train, pred_train) #1.0

'''Confusion Matrix'''
from sklearn.metrics import confusion_matrix
import pylab as pl
import matplotlib.pyplot as plt
labels = ['sales','service']
cm = confusion_matrix(y_test, pred, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
pl.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pl.xlabel('Predicted')
pl.ylabel('True')
pl.show()
'''Pandas Confusion Matrix'''
from pandas_confusion import ConfusionMatrix
cfm = ConfusionMatrix(y_test, y_pred)
print cfm
cfm.print_stats()
