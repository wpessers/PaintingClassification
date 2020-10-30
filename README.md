## Painting classification
This project is aimed at creating a deeplearning algorithm to classify paintings of 4 different painters, specifically:
* Brueghel
* Mondriaan
* Picasso
* Rubens

## Motivation
This project is the result of a coursework assignment, where the goal was to develop an algorithm that could differentiate between **2** painters. This project takes it up a notch by being able to discern between **4** painters.

## Results
#### Iteration 1
After making a deep convolutional neural network to distinguish between paintings from Brueghel and Mondriaan, I decided to expand the model to make work for 4 different painters.
At first I tried this with a model using the pretrained ResNet-50 convolutional neural network, with some Dense layers added on and ending with a relu and softmax layer.
The softmax layer enables multiclass classification. In this iteration I used the SGD optimizer because it performed well in the earlier binary classification version.

The model summary looked like this:

<img src="https://i.imgur.com/9Nq47zr.png" width=600>

Looking at the training and validation accuracy/loss the result isn't too convincing:

<img src="https://i.imgur.com/5I3rEpL.png" width=400>

The accuracy graph tells us there might be some overfitting, but overall the algorithm doesn't perform well either.

---
#### Iteration 2
For the next version I swapped the ResNet-50 model for the VGG16 pretrained model. I added a dropout layer to try to prevent overfitting.

<img src="https://i.imgur.com/GlSAB8t.png" width=600>

The results looked a little more promising this time around, with a validation accuracy of approximately 90% after some 30 epochs of training. However, one look at the confusion matrices shows us different:

<img src="https://i.imgur.com/xuwGzY9.png" width=400> <img src="https://i.imgur.com/13dzN0q.png" width=600>
We can see Bruegel and Mondriaan are rarely/never predicted correctly, respectively 2 Bruegel paintings and 0 Mondriaan paintings were predicted correctly. This was largely due to our dataset being unbalanced. Bruegel and Mondriaan where very underrepresented compared to the other 2 painters.

---
#### Iteration 3
To improve the algorithm I first redistributed the paintings accross the train, validation and test datasets to achieve 60% training data, 20% validation data and 20% test data. Then I added class weights to account for over- or underrepresented painters. The result was a lot better, as the confusion matrices show:

<img src="https://i.imgur.com/ramWbOM.png" width=600>

---
#### Iteration 4
In this iteration of the algorithm I noticed some weird patterns in the loss. The training loss exceeded the validation sometimes, which means we had been underfitting a little bit. To solve this I increased the complexity of the model by adding a few layers and changing the VGG16 pretrained model to XCeption. This increases the parameters our algorithm uses and actually solved the underfitting completely.

----
#### Iteration 5
This is the final version of the algorithm. To get good results I first had to try to solve one final overfitting issue. Looking at the validation and loss functions, the graph for our loss function shows the validation loss stalling and even increasing a little bit when training loss kept decreasing / stayed low:

<img src="https://i.imgur.com/f80xQ6v.png" width=400>

I noticed I accidentally removed my dropout-layer so I added it back in. This solved the problem to some extent. To improve the algorithm and generalise it even more I added in some data augmentation, which had a noticeable, positive effect on the results. I also swapped out the SGD optimizer for Adam, because SGD is supposedly outdated. This did improve the convergence speed of the algorithm a fair bit. I finally also added a ReduceLROnPlateau-callback and tuned the hyperparameters. I'm very happy with how it turned out, here's the result:

<img src="https://i.imgur.com/lWXm87Z.png" width=500> <img src="https://i.imgur.com/pdrh49W.png" width=500>
The reason for the relatively low accuracy on Mondriaan paintings is caused by two things I believe:

1. Low amount of available data, other painters had (sometimes significantly) more paintings.
1. Mondriaan's drastic painting style changes throughout his lifetime.

This concludes my endeavor in trying to make an AI art-connoiseur. This is what the final model looked like:

<img src="https://i.imgur.com/mYZvStY.png" width=600>

## License
 
The MIT License (MIT)

Copyright (c) 2020 Warre Pessers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
