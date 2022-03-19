import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

print('a')


labels_dict={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}
print('a')

size = 4
print('a')
# We load the xml file
classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
print('b')
print('c')
img_height = 200
img_width = 200
model=load_model("./model.h5")
testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './manotest',
)

class_names = testing_ds.class_names
plt.figure(figsize=(20, 20))
for images, labels in testing_ds.take(1):
    predictions = model.predict(images)
    print(predictions)
    predlabel = []
    
    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
    
    for i in range(40):
        ax = plt.subplot(10, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title('Predicted label:'+ predlabel[i])
        plt.axis('off')
        plt.grid(True)
