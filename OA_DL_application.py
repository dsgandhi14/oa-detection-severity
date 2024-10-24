#Importing Required Dependencies
print("Welcome to Deep Learning Application for detecting presence and grading severity of Osteoarthritis in Knee joint.")
print("Developed by Danish Gandhi")
print("Please wait while we load required modules.")
print(".......")
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.nn import softmax, sigmoid
from PIL import Image
import os
print("Module loading Successful")

#Accepting Input
print("Ensure image is square for best results")
print("Ensure image and app are in the same directory")
xrayName=str(input("Please enter X-Ray Image name: "))
imgarray=np.array(Image.open(xrayName).convert("RGB").resize((224,224)))
imgarray=imgarray.reshape((-1,224,224,3))
preprocess_input(imgarray)
grading_dict={0:"Healthy knee image.",1:"Doubtful joint narrowing with possible osteophytic lipping.",
              2:"(Minimal) Definite presence of osteophytes and possible joint space narrowing.",3:"(Moderate) Multiple osteophytes, definite joint space narrowing, with mild sclerosis.",
              4:"(Severe) Large osteophytes, significant joint narrowing, and severe sclerosis."}
#Model Architecture Definition (Severity Grading)
vgg=VGG16(input_shape=(224,224,3),weights="imagenet",include_top=False)
final_output=Flatten()(vgg.output)
final_output=Dense(units=512,activation ='leaky_relu')(final_output)
final_output=Dense(units=512,activation ='leaky_relu')(final_output)
final_output=Dense(units=5,activation="linear")(final_output)
model=Model(inputs=vgg.input,outputs=final_output)
model.load_weights("categorical.weights.h5")
#Model Architecture Definition (Presence Detection)
vgg_p=VGG16(input_shape=(224,224,3),weights="imagenet",include_top=False)
final_output_p=Flatten()(vgg_p.output)
final_output_p=Dense(units=512,activation ='leaky_relu')(final_output_p)
final_output_p=Dense(units=512,activation ='leaky_relu')(final_output_p)
final_output_p=Dense(units=1,activation="linear")(final_output_p)
model_p=Model(inputs=vgg_p.input,outputs=final_output_p)
model_p.load_weights("binary.weights.h5")
presence=(sigmoid(model_p.predict(imgarray))>0.5)
if presence:
    print("Osteoarthritis Detected")
    severity=np.argmax(softmax(model.predict(imgarray)))
    print(grading_dict[severity])
else:
    print("Healthy Knee Joint")
