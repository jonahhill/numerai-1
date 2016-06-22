#!/usr/bin/env python

import time
import csv
import pandas as pd
from sklearn import datasets, svm
from sklearn.externals import joblib


train_data = 'data/numerai_training_data.csv'
tournament_data = 'data/numerai_tournament_data.csv'

train_data = pd.read_csv(train_data)
dm = train_data.as_matrix()

train_samples = dm[:, 0:21]
train_labels = dm[:, 21:22]
train_labels = train_labels.reshape(len(train_labels),)


tournament_data = pd.read_csv(tournament_data)
cols = list(tournament_data.columns)
cols.remove('t_id')
t_id = tournament_data.as_matrix(['t_id'])
tm = tournament_data.as_matrix(cols)

def train():
  classifier = svm.SVC(gamma=0.001, probability=True)
  classifier.fit(train_samples, train_labels)
  joblib.dump(classifier, 'model/model_' + str(int(time.time())) + '.pkl')


def predict():
  classifier = joblib.load('model/model_1466591109.pkl')

  f = open('data/predictions.csv', 'wb')
  writer = csv.writer(f)
  t_id_list = list(t_id.reshape(len(t_id),))
  result_list = list(classifier.predict_proba(tm)[:,1:].reshape(len(tm),))

  rows = zip(t_id_list, result_list)
  for row in rows:
    writer.writerow(row)

  f.close()

  # k = 0
  # total = len(tm)
  # for i in tm:
  #   print str(k), '\t', classifier.predict_proba(i.reshape(1, -1))[0][1], '\t', str((float(k)/total)*100) + '%'
  #   k += 1
    
  # results = classifier.predict(tm)

if __name__ == '__main__':
  # train()
  predict()
