import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from scrape import *
# Permanently changes the pandas settings
pd.set_option('display.max_rows', 37)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

base_url = 'https://transcripts.cnn.com/'

print("reading data...")
#Read the data
df = pd.read_csv('news.csv')

#Get shape and head
df.shape
df.head()

#DataFlair - Get the labels
labels=df.label
labels.head()

print("Splitting dataset...")
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'],
                    labels, test_size=0.2, random_state=7)
#print("                 XTest set before fitting and transformation\n",x_test)

print("Initiallizing stopwords...")
#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

print("Fitting and transforming the data...")
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)
#print("---Transformed test set:\n",tfidf_test)

print("Initializing passive aggresive classifier...")
#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

print("Predicting on test set and calculating accuracy...")
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
#print("Test set:",y_test)
print(f'Accuracy: {round(score*100,2)}%')

#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

print("\nGetting links...")
#get links
links = get_links(base_url)

print("Scraping links...")
#traverse and scrape each link
content = []
for link in links:
    html = urllib.request.urlopen(link).read()
    text = text_from_html(html)
    #print(text)
    content.append(text)

#parse text and create dataframe
print("Parsing text and creating dataframe...")
dataframe = pd.DataFrame(content)
dataframe.columns = ['text']
print(dataframe['text'])

# transforming new data
print("Transforming new data...")
cnn_data = tfidf_vectorizer.transform(dataframe['text'])
#print("CNN data",cnn_data)

# predicting on trained model
print("Calculating final prediction...")
final_prediction = pac.predict(cnn_data)
print(len(final_prediction), "\n",final_prediction)

#updating dataframe with new predictions
print("updating dataframe...")
dataframe.insert(0, "New predictions", final_prediction)
print(dataframe)
