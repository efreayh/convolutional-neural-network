# Convolutional Neural Network
In this project, I created a modular convolutional neural network from scratch that is able to achieve 99% accuracy on the [MNIST data set]. The project was designed to be modular, allowing for experimentation with different neural network architectures similar to how neural networks can be created in machine learning libraries such as Keras.

![Convolutional Neural Network](https://d29g4g2dyqv443.cloudfront.net/sites/default/files/pictures/2018/convolutional_neural_network.png "Convolutional Neural Network")
*<center><sup>Diagram of a Convolutional Neural Network ([Image Source])</sup></center>*

## Training the Network
To train the network, clone the repository and place a csv containing the MNIST data set in a subfolder called "data". Then, run the following in the root directory:
```
make
./build/main
```

## Sample Output
```
Loading data set...
Starting training...
************ Epoch 1/10 ************
Training iteration: 56000/56000
Predicting...
Accuracy: 98.1143% Time: 619334ms
************ Epoch 2/10 ************
Training iteration: 56000/56000
Predicting...
Accuracy: 98.3429% Time: 619641ms
************ Epoch 3/10 ************
Training iteration: 56000/56000
Predicting...
Accuracy: 98.7143% Time: 614214ms
************ Epoch 4/10 ************
Training iteration: 56000/56000
Predicting...
Accuracy: 98.8143% Time: 617566ms
************ Epoch 5/10 ************
Training iteration: 56000/56000
Predicting...
Accuracy: 98.7929% Time: 627805ms
************ Epoch 6/10 ************
Training iteration: 56000/56000
Predicting...
Accuracy: 98.9357% Time: 631774ms
************ Epoch 7/10 ************
Training iteration: 56000/56000
Predicting...
Accuracy: 98.8857% Time: 610428ms
************ Epoch 8/10 ************
Training iteration: 56000/56000
Predicting...
Accuracy: 98.9357% Time: 643387ms
************ Epoch 9/10 ************
Training iteration: 56000/56000
Predicting...
Accuracy: 98.9786% Time: 653492ms
************ Epoch 10/10 ************
Training iteration: 56000/56000
Predicting...
Accuracy: 98.9643% Time: 676633ms
```

[MNIST data set]: https://en.wikipedia.org/wiki/MNIST_database
[Image Source]: https://developer.nvidia.com/discover/convolutional-neural-network