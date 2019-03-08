import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import _pickle as cPickle

def save(clf, name):
    with open(name, 'wb') as fp:
        cPickle.dump(clf,fp)
        print("saved")

def make_dict():
    direc = 'C:\\Users\\Administrator\\Desktop\\enron1\\emails\\'
    files = os.listdir(direc)
    # list comprehension to get all the emails from the files
    emails = [direc + email for email in files]

    #Creating a list of words per email
    words = []

    for email in emails:
        with open(email,encoding="iso8859_1") as f:
            blob = f.read()
            words += blob.split(' ')
    #Most repeated words
    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]
    return dictionary.most_common(3000)


#Preparing dataset into machine learning
#Feature Vectors

def make_dataset(dictionary):
    direc = 'C:\\Users\\Administrator\\Desktop\\enron1\\emails\\'
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    feature_set = []
    labels = []
    c = len(emails)

    for email in emails:
        data = []
        with open(email,encoding="iso8859_1") as f:
            words = f.read().split(' ')
            for entry in dictionary:
                data.append(words.count(entry[0]))
            feature_set.append(data)

            if "ham" in email:
                labels.append(0)
            if "spam" in email:
                labels.append(1)
    return feature_set, labels


d = make_dict()
features, labels = make_dataset(d)
x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2) # 80% for data training
#clasifier
clf = MultinomialNB()
clf.fit(x_train,y_train)

preds = clf.predict(x_test)
print(accuracy_score(y_test, preds))
save(clf, "text-classifier.mdl")




while True:
    features = []
    inp = input(">").split()
    if inp[0] == "exit":
        break
    for word in d:
        features.append(inp.count(word[0]))
    res = clf.predict([features])
    print (["Not Spam", "Spam!"][res[0]])
    
'''    
This is not my code. I simply took this tutorial https://www.youtube.com/watch?v=6Wd1C0-3RXM
and made it for Python 3 instead as its original for Python 2. 
I went over it to help myself understanding the vectorization patterns of Naive Bayes. 
'''
