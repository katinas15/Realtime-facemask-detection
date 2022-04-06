import cv2
import numpy as np
from tensorflow.keras.models import load_model

input_shape = 224

labels_dict={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
# webcam = cv2.VideoCapture(0) #Use camera 0
webcam = cv2.VideoCapture("../0319/su_kauke1.mp4") #Use video file
# webcam = cv2.VideoCapture("../0321/vid.mp4") #Use video file

# We load the xml file
classifier = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
# model=load_model("./model.h5")
model=load_model("./model.h5")
# model=load_model("./0319/model.h5")
counter = 30

while True:
    (rval, im) = webcam.read()

    if counter%30==0:

        im=cv2.flip(im,1,1) #Flip to act as a mirror

        # Resize the image to speed up detection
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

        # detect MultiScale / faces 
        faces = classifier.detectMultiScale(mini)

        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            #Save just the rectangle faces in SubRecFaces
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(input_shape,input_shape))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,input_shape, input_shape,3))
            reshaped = np.vstack([reshaped])
            result=model.predict(reshaped)
            print(result)
            
            label=np.argmax(result,axis=1)[0]
        
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
        # Show the image
        cv2.imshow('LIVE',   im)
        key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break

    counter += 1

# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()