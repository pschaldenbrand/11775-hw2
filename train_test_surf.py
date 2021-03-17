#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
import sys
import pdb
import time

# Train SVM

list_videos = 'labels/trainval.csv'
feat_dirs = ['surf_bof']#['resnet_avgpool_feat', 'resnet_3d_feat', 'r2plus1d_18_feat', 'mc3_18_feat']
output_file = 'models/surf.model'#'models/best.model'

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

layer_shapes = {}
mfcc_shape = {}

for line in fread.readlines()[1:]:
  video_id = line.strip().split(",")[0]
  feat = np.array([])

  label_list.append(int(df_videos_label[video_id]))

  # Add features
  for feat_dir in feat_dirs:
    feat_filepath = os.path.join(feat_dir, video_id + '.csv')
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


start_time = time.time()

y = np.array(label_list)
X = np.array(feat_list)

# # Augment data
# y = np.tile(y, factor)
# X = np.tile(X, (factor,1))

# X[len(label_list):,:] += (np.random.randn(len(X) - len(label_list), X.shape[1])*.1)

clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", \
    max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=3)
clf.fit(X, y)


# save trained SVM in output_file
pickle.dump(clf, open(output_file, 'wb'))
print('Elapsed Time ', time.time() - start_time, ' seconds')
print('training accuracy: ', accuracy_score(y, clf.predict(X)))


# Now test the model


start_time = time.time()
# 1. load mlp model
mlp = pickle.load(open(output_file, "rb"))

# 2. Create array containing features of each sample
fread = open('labels/test_for_student.label', "r")
feat_list = []
video_ids = []
for line in fread.readlines():
  # HW00006228
  video_id = os.path.splitext(line.strip())[0]
  video_ids.append(video_id)

  feat = np.array([])
  # Add features
  for feat_dir in feat_dirs:
    feat_filepath = os.path.join(feat_dir, video_id + '.csv')
    if not os.path.exists(feat_filepath):
      feat = np.concatenate((feat, np.zeros(256)))
    else:
      feat = np.concatenate((feat, np.loadtxt(feat_filepath)))
  feat_list.append(feat)

X = np.array(feat_list)

start_time = time.time()
# 3. Get predictions
# (num_samples) with integer
pred_classes = mlp.predict(X)

# 4. save for submission
with open('surf.csv', "w") as f:
  f.writelines("Id,Category\n")
  for i, pred_class in enumerate(pred_classes):
    f.writelines("%s,%d\n" % (video_ids[i], pred_class))

print('test run time', time.time() - start_time, 'seconds')