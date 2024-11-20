import numpy as np
import os
import time
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from PIL import Image

# Loading the training data
PATH = os.getcwd()
data_path = 'your_data_path'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        img_path = data_path + '/'+ dataset + '/'+ img      
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = x/255
        img_data_list.append(x)

img_data = np.array(img_data_list)
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
print (labels.shape)

step = 1400  # number of AF train image
step1 = 1600  # number of Normal train image
step2 = 1240  # number of ST train image
step3 = 1600  # number of VF train image
labels[step*0:step-1]=0
labels[step:step+step1-1]=1
labels[step+step1:step+step1+step2-1]=2
labels[step+step1+step2:step+step1+step2+step3-1]=3

names = ['AF','Normal','ST','VF'] 

Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

#########################################################################################
image_input = Input(shape=(224,224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

for layer in custom_vgg_model.layers[:-1]: 
	layer.trainable = False

custom_vgg_model.layers[2].trainable
# conf. weight name 
checkpoint = ModelCheckpoint(filepath='your_model_name.hdf5', 
            monitor='loss', 
            mode='min', 
            save_best_only=True)

custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


t=time.time()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=30, epochs=200, verbose=1, validation_data=(X_test, y_test), callbacks=[checkpoint])
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=30, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


####################################################################################################################
#%%
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(200)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.title(' ')
plt.grid(True)
plt.legend(['train','validation'])
plt.plot()

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('number of epochs')
plt.ylabel('accuracy')
plt.title(' ')
plt.grid(True)
plt.legend(['train','validation'],loc=4)
plt.plot()

#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_predic = custom_vgg_model.predict(img_data)
conMatrix = confusion_matrix((np.argmax(Y,axis=1)), np.argmax(y_predic,axis = 1), normalize = 'true')
conMatrixn = confusion_matrix((np.argmax(Y,axis=1)), np.argmax(y_predic,axis = 1))
print(conMatrix)
print(conMatrixn)
disp= ConfusionMatrixDisplay(conMatrix)
disp.plot()
dispn= ConfusionMatrixDisplay(conMatrixn)
dispn.plot()
