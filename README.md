# oa-detection-severity
Osteoarthritis Detection and Severity Grading using Deep Learning by Danish Gandhi

# Osteoarthritis Detection and Severity Grading using Deep Learning
This is an application to examine the knee joint for presence of osteoarthritis and grade its severity (KL grading) using xray imaging as input. This project employs deep learning, specifically a convolutional nueral network based on VGG16 architecture. The final generalization accuracies of 83.6% (AUC score 0.91) for detection and 69.8% for severity grading were achieved by using this methodology.

## Methodology 
Specification of Model used -
* VGG16 Model loaded with imagenet weights
* 2 fully connected Dense Layers with 512 neurons each and leaky relu activation
* Dense layer with 5 neurons and softmax activation
* Trained using Sparse Categorical Cross Entropy loss and Adam Optimizer
* Employed Data Augmentation for regularization

Different models were trained for 10 to 25 epochs while keeping different number of VGG16 layers trainable. Best accuracy was achieved for the model specified above. A Similar model trained using Binary Cross Entropy loss and a single sigmoid neuron in last layer was able to improve detection accuracy to 83.6% (AUC Score 0.91).
Dataset used : https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity
## Requirements

* Latest Version of Miniconda : https://docs.anaconda.com/miniconda/miniconda-install/
* Create a new conda environment
```
conda create --name oadetect python=3.9
```
* Activate conda environment
```
conda activate oadetect
```
* Install CUDA cuDNN with conda (For Nvidia GPU Support)
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
* Install Tensorflow 2.10 (Windows Native GPU Support)
```
pip install "tensorflow<2.11"
```
* Downgarde numpy to work with Tensorflow 2.10
```
pip uninstall numpy
```
```
pip install "numpy<2"
```
## How to Use
* Clone repository or download OA_DL_application to your local machine.
* Get weights from drive link: https://drive.google.com/drive/folders/1w7oDRkv0vq9xstXKqXL0TnidLIdNuVUV?usp=sharing
* Ensure weights are placed in same folder as the application.
* Navigate to folder with application and weights.
* Activate conda environment.
```
conda activate oadetect
```
* Execute OA_DL_application.py


