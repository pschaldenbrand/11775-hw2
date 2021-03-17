# 11-775 Homework 2

Peter Schaldenbrand <br/>
11-775 Spring 2021<br/>
Homework 2 <br/>
Andrew ID: pschalde <br/>
GitHub ID: pschaldenbrand <br/>
Kaggle ID: pittsburghskeet

## Surf Features

I extracted all frames from videos ahead of time using ```extract_frames.sh``` to speed up the feature extraction times.  It took 2 hours and 55 minutes to extract the surf features.  Selecting the frames took 25 minutes and training kmeans with 256 means took 6681 minutes.  Getting the bag of features took 2311 seconds.  I then trained an MLP with one hidden layer with 100 hidden units in it.  I used 5-fold cross validation and using the bag of features resulted in an average validation accuracy of 46.2%.  The model took 104 seconds to train for all 5 folds.

## CNN Features

It's important to normalize the images and ensure that they are properly shaped before feeding them into a pretrained model.  I followed the directions on torchvision.model's website for this process.  I used the resnet18 pretrained model at the avgpool layer to extract features.  I took the mean across the variable length dimension to produce a feature vector for size 512.  Training a 100 hidden unit, single layer MLP on this yielded 87% validation accuracy in 5-fold cross validation.

## Model Size Test

I knew from homework 1 that the MLP was far better at taking features and classifying the data, so I started this homework with the MLP.  Using the resnet18 avgpool layer features only, I searched for which hidden layer configuration may give me the best results.  To determine best results, I used 5-fold cross validation and average validation accuracy as a metric.  I test a single hidden layer MLP with 10, 100, and 1000 hidden units, then tested an mlp with two hidden layers with 100 and 100 hidden units and 1000 and 1000 hidden units.

It was difficult to determine which configuration gave the best results since many validation accuracies were close.  A single layer 100 hidden unit model produced 87.23% validation accuracy, but two layer models with two 1000 layers and a single layer with 1000 units also got around 87% accuracy. Folowing occam's razor, I decided that the single layer with 100 hidden unit model was best since it was simplest with good results.

## Video Features

I used the 'r3d_18' model which is designed to classify videos from the torchvision.models package to extract features.  Just like with resnet18 I extracted from the avgpool layer at the end of the model then averaged them across the variable length dimension. This produced a vector of size 512 values for features.

When I used the r3d_18 features in addition to the resnet18 featrures (by concatenating them), I saw a huge boost in validation accuracy when performing 5-fold cross validation.  resnet18 features alone produce 87.2% and adding the r3d_18 features increased it to 94.8% validation accuracy.  Actually, using the r3d_18 features alone got a 94.7% accuracy, so it appears that the resnet18 features do not aid it much.

With the success of the r3d_18 model, I thought I'd try the r2plus1d_18 model which torchvision totes as having higher accuracy for action recognition.  I extract the features in the same way as with r3d_18 and the feature vector was of size 512. Adding the r2plus1d_18 features, the 5-fold cross validation accuracy increased to 97.2%.  I added the mc3_18 model's features too in the same way for a small increase in accuracy.

I extracted all of the images ahead of time so the feature extraction would be faster.  Each of these feature extractions took between 3 and 5 hours to complete.

## Data Augmentation

I tried to augment the data since there really isn't a ton of training samples to work with.  My augmentation scheme constisted of duplicating data, then adding noise to the duplicated data.  The average of the featuers was 0.78 and the standard deviation was 0.739.  I tested different amounts of duplication and different amounts of noise to add.  The noise was gaussian noise with a set standard deviation and zero mean.  The best noise to add had a 0.2 standard deviation. Data augmentation at most increased the validation accuracy by 0.2%.  This difference didn't seem significant enough for me to use this in my best model as I wanted to make the model as simple as possible to increase generalization.

## Best Model

My best model used the features extracted using the resnet18, r3d_18, mc3_18, and r2plus1d_18 models.  The model was a single layer with 100 hidden unit MLP which used early stopping.  The model only took 9.8 seconds to train!  Running the model on the test set took 188.78 seconds, though most of that time was for loading the data. Training the model only took 7.9 seconds and running it on the test set took 0.05 seconds.


## Running My Code

Place the videos into a directory called ```videos```.

Extract frames

```extract_frames.sh```

Extract features

```
python cnn_feat_extraction.py videos resnet_avgpool_feat
python 3d_cnn_feat_extraction.py videos resnet_3d_feat
python 3d_cnn_feat_extraction.py videos r2plus1d_18_feat --model r2plus1d_18
python 3d_cnn_feat_extraction.py videos mc3_18_feat --model mc3_18
```

Train model and save predictions:

```
python train_best.py
```

## Confusion Matrix

The 5-fold cross validation confusion matrix follows, where the row indicates the true value and the column indicates the predicted value.

|                             | dribbling <br>basketball | mowing <br>lawn | playing <br>guitar | playing <br>piano | playing <br>drums | tapping<br> pen | blowing <br>out <br>candles | singing | tickling | shoveling <br>snow |
|-----------------------------|--------------------------|-----------------|--------------------|-------------------|-------------------|-----------------|-----------------------------|---------|----------|--------------------|
| dribbling <br>basketball    | 597                      | 1               | 0                  | 0                 | 1                 | 0               | 0                           | 0       | 1        | 1                  |
| mowing <br>lawn             | 0                        | 599             | 1                  | 0                 | 0                 | 0               | 0                           | 0       | 0        | 1                  |
| playing <br>guitar          | 0                        | 1               | 579                | 1                 | 10                | 0               | 1                           | 7       | 1        | 1                  |
| playing <br>piano           | 1                        | 0               | 1                  | 500               | 0                 | 0               | 2                           | 5       | 0        | 0                  |
| playing <br>drums           | 0                        | 0               | 6                  | 2                 | 583               | 2               | 1                           | 7       | 0        | 0                  |
| tapping<br> pen             | 0                        | 0               | 0                  | 1                 | 1                 | 520             | 0                           | 2       | 1        | 1                  |
| blowing <br>out <br>candles | 0                        | 0               | 1                  | 0                 | 0                 | 0               | 592                         | 3       | 5        | 0                  |
| singing                     | 1                        | 1               | 12                 | 5                 | 11                | 0               | 7                           | 556     | 6        | 2                  |
| tickling                    | 0                        | 1               | 1                  | 0                 | 1                 | 3               | 4                           | 3       | 407      | 0                  |
| shoveling <br>snow          | 1                        | 1               | 0                  | 0                 | 1                 | 0               | 0                           | 0       | 0        | 598                |

My model is generally very good.  It's accuracy is >97% on the validation data when doing 5-fold cross validation but it still makes some mistakes.  Notably, playing guitar was predicted to be playing drums 10 times and the reverse 6 times.  The model often predicts singing when the video isn't of singing.  The model also fails to correctly predict singing often. This makes sense since singing looks visually similar to many of the other actions that could be here.  Without hearing the voice, it can be difficult to determine if someone is singing.
