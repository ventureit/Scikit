from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

#twenty_train.target_names
len(twenty_train.data)
len(twenty_train.filenames)
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])
twenty_train.target[:10]

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

#index value of a word in the vocabulary is linked to its frequency in the whole training corpus
count_vect.vocabulary_.get(u'alogorithm')
count_vect.vocabulary_.get(u'newspaper')
count_vect.vocabulary_.get(u'horrifying')
count_vect.vocabulary_.get(u'impossible')

#term frequencies - number of occurences/total number of words: tf
#downscale weights for words that occur in many documents in the corpus (less informative): tf-idf (Term Frequency times Inverse Document Frequency)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_train_tfidf.shape

#Train a classifier
#Naive Bayes multinormal variant
#Multinomial distribution - multinomial distribution gives the probability of any particular combination of numbers of successes for the various categories
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

#Predict the outcome on a new document (Try to predict the category of a post)
#Use transform instead  of fit_transform (already fitted to the training set)
docs_new = ['evolution', 'devil']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)


predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

#zip combines data into series (zips the data into a series)
print(set(zip(docs_new, predicted)))

#Build a pipeline for vectorize => transformer => classifier
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

#Test the accuracy of the model
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

#Test the accuracy of a different model - Support Vector Machine
