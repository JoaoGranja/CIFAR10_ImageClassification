# CIFAR10_ImageClassification

## *Project Description*
In this project, a step by step process will be developed for the Image Classification task over the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Image classification is the process of a model classifying input images into their respective category classes.  In this Image Classification task, not only will I design and develop a simple and efficient CNN model from scratch but also learn how to implement a pre-trained model. When pre-trained model is used, transfer learning and fine-tuning techniques are applied.
I will also compare the performance of the two approaches looking on loss and accuracy metrics over the testing dataset.

## *Dataset*
[CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) will be used which is comprised of 60000 32x32 color images in 10 classes, with 6000 images per class. It is considered as one of the best dataset to start working with Image Classification task. Actually, the small image resolution and the large number of images make it a great tool for quickly tuning a machine learning model. 

## *Project steps*
The [image_classification_main.ipynb](https://github.com/JoaoGranja/CIFAR10_ImageClassification/blob/master/image_classification_main.ipynb) script was created to train and evaluate the CNN models over the CIFAR10 dataset. In this project, I took the following steps:

<ul>
  <li><strong>Colab preparation</strong> - In this step,  all necessary packages/libraries are installed and my google drive account is shared.</li>
  <li><strong>Configuration and Imports</strong> - All modules are imported and the 'args' dictionary is built with some configuration parameters. </li>
  <li><strong>Loading the dataset</strong> - CIFAR10 dataset is loaded and the data are analysed. </li>
  <li><strong>Data pre-processing and data augmentation</strong> - Apply one hot enconding for the output class and some simple data augmentation is used. </li>
  <li><strong>Optimizer</strong> - Choose the optimizer for model training </li>
  <li><strong>Model</strong> - Based on 'args' configuration, make the model. The model architecture is built on models module. </li>
  <li><strong>Training</strong> - The training process runs in this step. Several callbacks are used to improve the trainig process. </li>
  <li><strong>Visualize models result</strong> - After the model is trained, the accuracy and loss of the model is plotted.</li>
  <li><strong>Evaluation</strong> - After all models are trained, the evaluation over a testing dataset is done. </li>
</ul>

## *Conclusion*
In sume, all models achieved more than **80%** of accuracy over the testing dataset, with VGG16 model achieved the highest score (more than **83%**).
The goal of this project was not to achieve the highest accuracy score over the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) but to approach the Image Classification task. For better results, several future works are usefull (use better data augmentation techniques, apply regularization techniques to reduce overfitting, tune hyperparameters of the model) 
