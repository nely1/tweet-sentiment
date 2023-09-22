import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from collections import Counter
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


train_data = pd.read_csv("Train.csv", sep=',')
test_data = pd.read_csv("Test.csv", sep=',')

# split the training data into 8:2 ratio
X_train_train_data, X_test_train_data, Y_train_train_data, Y_test_train_data = train_test_split(train_data[['text']], 
                                                                                                train_data[['sentiment']],
                                                                                                test_size=0.2,
                                                                                                random_state=42)

# print(X_train_train_data)
# print(Y_train_train_data)
# print(X_test_train_data)
# print(Y_test_train_data


# separating instance and label for Train
X_train_raw = [x[0] for x in X_train_train_data.values]
Y_train = [x[0] for x in Y_train_train_data.values]
Y_test = [x[0] for x in Y_test_train_data.values]


#separating instance and label for Test
X_test_raw = [x[0] for x in X_test_train_data[['text']].values]


# -------------------------Vectorise the words with Bag of Words--------------------------------------------------------

stop_words = stopwords.words('english')
BoW_vectorizer = CountVectorizer(stop_words = stop_words)

X_train_BoW = BoW_vectorizer.fit_transform(X_train_raw)

X_test_BoW = BoW_vectorizer.transform(X_test_raw)



#----------------------------Set the baseline accuracy with the zero-r baseline-----------------------------------------
zero_r = DummyClassifier(strategy='most_frequent')
zero_r.fit(X_train_raw, Y_train)

zr_pred = zero_r.predict(X_train_raw)
label_counter = Counter(Y_train)
label_counter.most_common()

zero_r_score = zero_r.score(X_train_raw, Y_train)
# print(zero_r_score)


#-----------------------------Implement chi2 feature selection----------------------------------------------------------

x2 = SelectKBest(chi2, k=500)

X_train_x2 = x2.fit_transform(X_train_BoW,Y_train)
X_test_x2 = x2.transform(X_test_BoW)



#----------------------------Stacking class-----------------------------------------------------------------------------
class StackingClassifier():

    def __init__(self, classifiers, metaclassifier):
        self.classifiers = classifiers
        self.metaclassifier = metaclassifier

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)
        X_meta = self._predict_base(X)
        self.metaclassifier.fit(X_meta, y)
    
    def _predict_base(self, X):
        yhats = []
        for clf in self.classifiers:
            yhat = clf.predict_proba(X)
            yhats.append(yhat)
        yhats = np.concatenate(yhats, axis=1)
        assert yhats.shape[0] == X.shape[0]
        return yhats
    
    def predict(self, X):
        X_meta = self._predict_base(X)     
        yhat = self.metaclassifier.predict(X_meta)
        return yhat
    def score(self, X, y):
        yhat = self.predict(X)
        return accuracy_score(y, yhat)
    
    
#------------------------------------Print accuracies for MNB and Logistic Regression-----------------------------------

models = [MultinomialNB(),
          LogisticRegression(max_iter = 1000)]
titles = ['MNB',
          'Logistic Regression']

lr_chi2 = []

mnb_chi2 = []

ks = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

for k in ks:
    print('\n--------------------------------------- K = ', k,'------------------------------')
    
    x2 = SelectKBest(chi2, k=k)

# BoW implemntation
 
    x2.fit(X_train_BoW,Y_train)
    X_train_x2 = x2.transform(X_train_BoW)
    X_test_x2 = x2.transform(X_test_BoW)

    Xs = [(X_train_x2, X_test_x2)]
    X_names = ['x2']
    for title, model in zip(titles, models):
        print('\n=========',title, '(with k=',k,'features): ')
        for X_name, X in zip(X_names, Xs):
            X_train_t, X_test_t = X
            model.fit(X_train_t, Y_train)
            y_test_predict = model.predict(X_test_t)
            accuracy =  accuracy_score(Y_test_train_data, y_test_predict)
            if title == 'Logistic Regression':
                lr_chi2.append(accuracy)
            else:
                mnb_chi2.append(accuracy)
            
            print(X_name, 'accuracy is:',  accuracy)

#-------------------------------Create a list of accuracies for the stacker---------------------------------------------

# Bag of words
stacker_acc = [] 
meta_classifier_lr = LogisticRegression(max_iter = 1000)
stacker_lr = StackingClassifier(models, meta_classifier_lr)

for k in ks: 
    
    x2 = SelectKBest(chi2, k=k)
    x2.fit(X_train_BoW,Y_train)
    X_train_x2 = x2.transform(X_train_BoW)
    X_test_x2 = x2.transform(X_test_BoW)
    
    for model in models:
        model.fit(X_train_x2, Y_train)
    stacker_lr.fit(X_train_x2, Y_train)
    accuracy = stacker_lr.score(X_test_x2, Y_test_train_data)
    stacker_acc.append(accuracy)

print("\n Stacker accuracies")
print(stacker_acc)


#---------------------------------Cross validation for stacking --------------------------------------------------------
print("\n Cross Validation for stacker")

total = 0

kf = KFold(n_splits=10, shuffle = False)
kf.get_n_splits(train_data)

for train_index, test_index in kf.split(train_data):
    training_data_x = train_data.iloc[train_index]['text'].values
    training_data_y = train_data.iloc[train_index]['sentiment'].values
    testing_data_x = train_data.iloc[test_index]['text'].values
    testing_data_y = train_data.iloc[test_index]['sentiment'].values
    
    cv_vectorizer = CountVectorizer(stop_words = stop_words)
    cv_x_train_BoW = cv_vectorizer.fit_transform(training_data_x)
    cv_x_test_BoW = cv_vectorizer.transform(testing_data_x)
    
    cv_x2 = SelectKBest(chi2, k=3965)
    cv_x_train_x2 = cv_x2.fit_transform(cv_x_train_BoW, training_data_y)
    cv_x_test_x2 = cv_x2.transform(cv_x_test_BoW)
    
    meta_classifier_lr = LogisticRegression(max_iter = 1000)
    stacker_lr = StackingClassifier(models, meta_classifier_lr)
    
    for model in models:
        model.fit(cv_x_train_x2, training_data_y)
    stacker_lr.fit(cv_x_train_x2, training_data_y)
    total = total + stacker_lr.score(cv_x_test_x2, testing_data_y)
    print(stacker_lr.score(cv_x_test_x2, testing_data_y))
                                    
average = total / 10

print(f'average = {average}')




