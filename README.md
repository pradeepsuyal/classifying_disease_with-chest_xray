![Apache License](https://img.shields.io/hexpm/l/apa)  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)    ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)   ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![Made with matplotlib](https://user-images.githubusercontent.com/86251750/132984208-76ce70c7-816d-4f72-9c9f-90073a70310f.png)   ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white) ![medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![udemy](https://img.shields.io/badge/Udemy-EC5252?style=for-the-badge&logo=Udemy&logoColor=white)

<p align="center">
  <img src="https://media.giphy.com/media/dt0KXLj7bzwZuRQBwY/giphy.gif" alt="animated" />
</p>

gif source : [giphy](https://giphy.com/gifs/dt0KXLj7bzwZuRQBwY)

## Operation Department

* AI/ML/DL has been revolutionizing healthcare and medicine:
    
      Medical imagery 
      Drug research 
      Genome development 

* Deep learning has been proven to be superior in detecting and classifying disease using imagery data.

* Skin cancer could be detected more accurately by Deep Learning than by dermatologists (2018). 

      Human dermatologists detection = 86.6%
      Deep Learning detection = 95%

`Reference: "Computer learns to detect skin cancer more accurately than doctors". The Guardian. 29 May 2018`

## Acknowledgements

 - [python for ML and Data science, udemy](https://www.udemy.com/course/python-for-machine-learning-data-science-masterclass)
 - [ML A-Z, udemy](https://www.udemy.com/course/machinelearning/)
 - [365 Data Science](https://learn.365datascience.com/career-tracks/data-scientist/)
 
## Appendix

* [Aim](#aim)
* [Dataset used](#data)
* [Run Locally](#run)
* [Exploring the Data](#viz)
   - [Matplotlib](#matplotlib)
* [solving the task](#fe)
* [prediction](#models)
* [conclusion](#conclusion)

## AIM:<a name="aim"></a>

Automate the process of detecting and classifying chest disease and reduce the cost and time of detection. 

## Dataset Used:<a name="data"></a>

The team has collected extensive X-Ray chest data and with the help of this dataset we can develop a model that could detect and classify the diseases in less than 1 minute. 

Dataset contain 133 images that belong to 4 classes: 
    
    Healthy 
    Covid-19
    Bacterial Pneumonia
    Viral Pneumonia 

## Run locally:<a name="run"></a>

Clone the project

```bash
https://github.com/pradeepsuyal/classifying_disease_with-chest_xray.git
```

Go to the project directory

```bash
  cd classifying_disease_with-chest_xray
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```

If you output `pip freeze` to a file with redirect >, you can use that file to install packages of the same version as the original environment in another environment.

First, output requirements.txt to a file.

```bash
  $ pip freeze > requirements.txt
```

Copy or move this `requirements.txt` to another environment and install with it.

```bash
  $ pip install -r requirements.txt
```

## Exploring the Data:<a name="viz"></a>

I have used matplotlib for basic visualization.

**Matplotlib:**<a name="matplotlib"></a>
--------
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, web application servers, and four graphical user interface toolkits.You can draw up all sorts of charts(such as Bar Graph, Pie Chart, Box Plot, Histogram. Subplots ,Scatter Plot and many more) and visualization using matplotlib.

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install matplotlib:

    pip: pip install matplotlib

    anaconda: conda install matplotlib
    
    import matplotlib.pyplot as plt

for more information you can refer to [matplotlib](https://matplotlib.org/) official site

## approach for making prediction<a name="fe"></a>
-------

* Used image generator to generate tensor images data and normalize them and used 20% of the data for cross-validation.
* Generate batches of 40 images and Performed shuffling and image resizing.
* Encoded Labels with to numemric values(0,1,2,3).
* imported model with pretrained weights.
* Building and Training Deep learning model.
* save the best model with lower validation loss.
* Evaluted Trained Deep learning model. 

## Prediction:<a name="models"></a>
------

**CONVOLUTIONAL NEURAL NETWORKS**

* The first CNN layers are used to extract high level general features. 
* The last couple of layers are used to perform classification (on a specific task).
* Local respective fields scan the image first searching for simple shapes such as edges/lines 
* These edges are then picked up by the subsequent layer to form more complex features.

![image](https://user-images.githubusercontent.com/86251750/146647568-0e5e1eeb-3ab7-47d6-b010-acb503ce6d7c.png)

![image](https://user-images.githubusercontent.com/86251750/146647609-90f4079d-fe25-445a-ab2f-c89c629033bc.png)

**Deep learning history**

There are many trained off the shelve convolutional neural networks that are readily available such as: 

    LeNet-5 (1998): 7 level convolutional neural network developed by LeCun that works in classifying hand writing numbers.
    AlexNet (2012): Offered massive improvement, error reduction from 26% to 15.3%
    ZFNEt (2013): achieved error of 14.8%
    Googlenet/Inception (2014): error reduction to 6.67% which is at par with human level accuracy.
    VGGNet (2014)
    ResNet (2015): Residual Neural Network includes “skip connection” feature and therefore enabled training of 152 layers without vanishing gradient issues. Error of 3.57% which is superior than humans. 

[source](https://medium.com/analytics-vidhya/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5)

**RESNET (RESIDUAL NETWORK)**
       
* As CNNs grow deeper, vanishing gradient tend to occur which negatively impact network performance.
* Vanishing gradient problem occurs when the gradient is back-propagated to earlier layers which results in a very small gradient. 
* Residual Neural Network includes “skip connection” feature which enables training of 152 layers without vanishing gradient issues. 
* Resnet works by adding “identity mappings” on top of the CNN. 
* ImageNet contains 11 million images and 11,000 categories. 
* ImageNet is used to train ResNet deep network.

![image](https://user-images.githubusercontent.com/86251750/146647728-cbc4a194-6cf6-4ffe-9f2f-c23ffe530f25.png)

**TRANSFER LEARNING**

* Transfer learning is a machine learning technique in which a network that has been trained to perform a specific task is being reused (repurposed) as a starting point for another similar task.
* Transfer learning is widely used since starting from a pre-trained models can dramatically reduce the computational time required if training is performed from scratch. 

![image](https://user-images.githubusercontent.com/86251750/146648910-1600aa56-abd9-4705-90b0-a7804bec3173.png)

photo credit : [link1](https://commons.wikimedia.org/wiki/File:Lillehammer_2016_-_Figure_Skating_Men_Short_Program_-_Camden_Pulkinen_2.jpg), [link2](https://commons.wikimedia.org/wiki/Alpine_skiing#/media/File:Andrej_%C5%A0porn_at_the_2010_Winter_Olympic_downhill.jpg)

* “Transfer learning is the improvement of learning in a new task through the transfer of knowledge from a related task that has already been learned”—Transfer Learning, Handbook of Research on Machine Learning Applications, 2009.
* In transfer learning, a base (reference) Artificial Neural Network on a base dataset and function is being trained. Then, this trained network weights are then repurposed in a second ANN to be trained on a new dataset and function. 
* Transfer learning works great if the features are general, such that trained weights can effectively repurposed.
* Intelligence is being transferred from the base network to the newly target network.

*TransferLearning process*

![image](https://user-images.githubusercontent.com/86251750/146647886-2d073768-2de5-4f6e-a086-2b3091903ca0.png)

*Why do we keep the First layer?*

* The first CNN layers are used to extract high level general features. 
* The last couple of layers are used to perform classification (on a specific task).
* So we copy the first trained layers (base model) and then we add a new custom layers in the output to perform classification on a specific new task.

![image](https://user-images.githubusercontent.com/86251750/146647947-315cae13-60d3-48ee-bf43-d06b6ade0660.png)

*TRANSFER LEARNING TRAINING STRATEGIES*

    - Strategy #1 Steps: 
         Freeze the trained CNN network weights from the first layers. 
         Only train the newly added dense layers (with randomly initialized weights).
    - Strategy #2 Steps: 
         Initialize the CNN network with the pre-trained weights 
         Retrain the entire CNN network while setting the learning rate to be very small, this is critical to ensure that you do not aggressively change the trained weights.

Transfer learning advantages are:
- Provides fast training progress, you don’t have to start from scratch using randomly initialized weights
- You can use small training dataset to achieve incredible results


**EVauting DeepLearning performance*

![download](https://user-images.githubusercontent.com/86251750/146648060-7c8781fc-e088-4c0c-a0b1-abf480c9637d.png)

![download](https://user-images.githubusercontent.com/86251750/146648063-253a2cc0-79a2-4893-9ddb-3058f8231183.png)

![download](https://user-images.githubusercontent.com/86251750/146648070-820cbe88-f5cd-4b4a-b2cd-dad4bf6e367c.png)

                 precision  recall  f1-score   support

           0       0.83      1.00      0.91        10
           1       0.80      0.80      0.80        10
           2       1.00      0.50      0.67        10
           3       0.77      1.00      0.87        10

    accuracy                           0.82        40

![download](https://user-images.githubusercontent.com/86251750/146648096-777c9e20-2ef8-4e62-8603-8b0f00ddf9e0.png)

## CONCLUSION:<a name="conclusion"></a>
-----
Training with deep learning model gives me quite good performance with accuracy of more than 80% and also provides precision,recall and f1 score greater than 50%. However since the dataset contains less than 1000 images, we didn't get higher precision and recall for some classes and thus we can't say that the model is best becuase of less training data.
If we want to further improve the performance then data collection should be needed because Deep learning models performs better if they get thousands and millions of images as its input

    NOTE--> we can further improve the performance by using other classification model such as CART model(XGBOOST, LIGHTGBM, CATBOOST, DecissionTree etc) and many more. Further performance can be improved by using various hyperparameter optimization technique such as optuna,hyperpot, Grid Search, Randomized Search, etc.   

![download](https://user-images.githubusercontent.com/86251750/146648494-a5c6f354-4433-42f1-b176-1296cb00f142.png)
