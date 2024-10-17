## Project description
This repository contains data preparation, model architecture, training and validation scripts of an image classifier that was developed as one of my university programming assignments. The project was built using [PyTorch](https://github.com/pytorch/pytorch) as a ML/DL framework. For the sake of convinience, [TensorBoard](https://github.com/tensorflow/tensorboard) was utilized to plot some important metrics (accuracy, loss, etc.) for supervising the training and evaluation phases.

The model is capable of classifying images into **20 different classes** (the exact classes can be found in the assignment PDF file).

To evaluate performance of trained models, we were provided with evaluation server. It took the trained model and tested it on a unknown set of images. This particular model could reach best **~0.77 accuracy** on a dedicated testing dataset.

The model architecture is a CNN derived from VGG16. In particular, the derived architecture has different from VGG16 input size, slightly altered hyperparameters, more/less convolutional, max pooling layers. VGG16 was choisen as a base architecture because it seemed a decent option for image classification tasks. Also, it was easier to implement in code than ResNet or similar architectures.

The training data was contributed by course participants, which was then cleaned by the university staff. After that, my task was to prepare, augment and load the data to disk/RAM/VRAM for further training and/or evaluation. 

The directory ```runs/``` contains tensorboard log files for different combinations of hyperparameters, model architecture, optimizers, training (augmented)/testing data, etc.

**Note:** this repository is effectively archived and is not planned to be further developed (except for probably text/code typos).

## How to run
To start training the model, create a directory ```training_data/``` with the train data and run ```train.py``` script with an arbitrary positional argument (for example, on Windows, ```python.exe train.py model```). ```train.py``` loads, splits, augments the data, and then trains, evaluates, tests, and saves the model. It also logs the process to the standard output and tensorboard log files.

If your machine has a dedicated GPU which supports CUDA, the whole training and evaluation procceses will done on the GPU. Otherwise, the CPU is going to be used.

To open TensorBoard dashboard for metric monitoring, you should have TensorBoard properly installed on the machine, and run the following in the terminal/cmd/etc.:
```tensorboard --logdir={path_to_project_dir}/runs```
where ```path_to_project_dir``` is to be substitued.

It's important to mention that the model accuracy strongly depeneds on the training data. For this particular architecture, augmentation/training/evaluation approach, the better and the more the data => the higher the accuracy.
