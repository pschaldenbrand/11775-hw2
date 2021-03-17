#!/bin/python

import os
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys
import time
import collections
import csv
import argparse
from tqdm import tqdm
import numpy as np
# Generate SURF-Bag-of-Word features for videos
# each video is represented by a single vector

parser = argparse.ArgumentParser()
parser.add_argument("kmeans_model")
parser.add_argument("surf_path")
parser.add_argument("cluster_num", type=int)
parser.add_argument("file_list")
parser.add_argument("output_path")

if __name__ == '__main__':
  args = parser.parse_args()

  # 1. load the kmeans model
  kmeans = pickle.load(open(args.kmeans_model, "rb"))

  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  # 2. iterate over each video and
  # use kmeans.predict(surf_features_of_video)
  start = time.time()
  fread = open(args.file_list, "r")
  for line in tqdm(fread.readlines()):
    surf_path = os.path.join(args.surf_path, line.strip() + ".p")
    bof_path = os.path.join(args.output_path, line.strip() + ".csv")

    if not os.path.exists(surf_path):
      continue

    # num_frames, (num_keypoints, 64)
    featurelist = pickle.load(open(surf_path, 'rb'))
    # 1. collect all features from all frames into a list
    X = []
    for i in range(len(featurelist)):
      if featurelist[i] is None:
        continue
      X.extend(featurelist[i])

    if not X:
      tqdm.write("warning, %s has empty features.." % line.strip())
      continue
    X = np.array(X)
    # (num_frames*num_keypoints,), each row is an integer for the clostest cluster center
    kmeans_result = kmeans.predict(X)

    dict_freq = collections.Counter(kmeans_result)
    # create dict containing 0 count for cluster number
    keys = np.arange(0, args.cluster_num, 1)
    values = np.zeros(args.cluster_num, dtype="float")
    dict2 = dict(zip(keys, values))
    dict2.update(dict_freq)
    list_freq = list(dict2.values())
    # normalize the frequency by dividing with frame number
    list_freq = np.array(list_freq) / len(featurelist)
    np.savetxt(bof_path, list_freq)

  end = time.time()
  print("K-means features generated successfully!")
  print("Time for computation: ", (end - start))
