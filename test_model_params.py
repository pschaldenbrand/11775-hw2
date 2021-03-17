#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
import sys
import pdb
import time


list_videos = 'labels/trainval.csv'
feat_dirs = ['resnet_avgpool_feat', 'resnet_3d_feat', 'r2plus1d_18_feat', 'mc3_18_feat']#['surf_bof']
folds = 5

factor = 1
std = 0.1

np.random.seed(0)

start_time = time.time()

# 1. read all features in one array.
fread = open(list_videos, "r")
feat_list = []
# labels are [0-9]
label_list = []
# load video names and events in dict
df_videos_label = {}
for line in open(list_videos).readlines()[1:]:
  video_id, category = line.strip().split(",")
  df_videos_label[video_id] = category

feat_shape = {}

for line in fread.readlines()[1:]:
  video_id = line.strip().split(",")[0]
  feat = np.array([])

  label_list.append(int(df_videos_label[video_id]))

  # Add features
  for feat_dir in feat_dirs:
    feat_filepath = os.path.join(feat_dir, video_id + '.csv')
    #feat = np.concatenate((feat, np.loadtxt(feat_filepath)))
    if not os.path.exists(feat_filepath):
      feat = np.concatenate((feat, np.zeros(256)))
    else:
      feat = np.concatenate((feat, np.loadtxt(feat_filepath)))


  feat_list.append(feat)

n = len(label_list)
inds = np.arange(n)
np.random.shuffle(inds)

label_list = np.array(label_list)
feat_list = np.array(feat_list)
print('feat_list shape ',feat_list.shape)

print('features stats', np.mean(feat_list), np.std(feat_list))

# for param in [(1000,1000,1000), (10000)]:
#for param in range(1,10):
# for param in [(100,), (100,100), (1000,), (1000,1000)]:
for param in [1]:
  #print(param)

  conf_mat = None


  all_val_acc = []
  all_train_acc = []


  for fold in range(folds):
    start_val = int(n * (float(fold)/folds))
    end_val = min(int(n * (float(fold+1)/folds)), n)

    train_fold_inds = np.concatenate((inds[:start_val], inds[end_val:]))
    val_fold_inds = inds[start_val:end_val]

    train_label_list = label_list[train_fold_inds]
    train_feat_list = feat_list[train_fold_inds]

    val_label_list = label_list[val_fold_inds]
    val_feat_list = feat_list[val_fold_inds]

    y = np.array(train_label_list)
    X = np.array(train_feat_list)

    factor = param
    if factor > 1:
      # Augment data
      y = np.tile(y, factor)
      X = np.tile(X, (factor,1))

      X[len(label_list):,:] += (np.random.randn(len(X) - len(label_list), X.shape[1])*.2)


    clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", \
        max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=4)
    clf.fit(X, y)

    y_val = np.array(val_label_list)
    X_val = np.array(val_feat_list)
    acc = accuracy_score(y_val, clf.predict(X_val))

    cf = confusion_matrix(y_val, clf.predict(X_val))
    conf_mat = cf if conf_mat is None else conf_mat + cf

    all_train_acc.append(accuracy_score(y, clf.predict(X)))
    all_val_acc.append(acc)


  print(conf_mat)

  # save trained SVM in output_file
  print('Elapsed Time ', time.time() - start_time, ' seconds')
  print('Average training accuracy: ', np.array(all_train_acc).mean())
  print('Average validation accuracy: ', np.array(all_val_acc).mean())

