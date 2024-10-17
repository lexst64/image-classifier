This repository contains data preparation, model architecture, training and validation scripts of an image classifier that was done as one of my university programming assignments. The project was built using PyTorch as a ML/DL framework. For the sake of convinience, tensorboard was utilized to plot some important metrics (accuracy, loss, etc.) for supervising the training and evaluation phases.

The model is capable of classifying images into 20 different classes (the exact classes can be found in the assignment PDF file).

To evaluate performance of trained models, we were provided with evaluation server. It took the trained model and tested it on a unknown set of images. This particular model could reach best ~0.77 accuracy.

The model architecture is a CNN derived from VGG16. In particular, the derived architecture has different from VGG16 input size, slightly altered hyperparameters, more/less convolutional, max pooling layers. VGG16 was choisen as a base architecture because it seemed a decent option for image classification tasks. Also, it was easier to implement in code than ResNet or similar architectures.

The training data was contributed by course participants, which was then cleaned by the university staff.

The directory ```runs/``` contains tensorboard log files for different combinations of hyperparameters, model architecture, optimizers, training (augmented)/testing data, etc.
