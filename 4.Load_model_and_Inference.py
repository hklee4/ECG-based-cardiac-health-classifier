import keras
import numpy as np
import os
import time
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

new_model = keras.models.load_model('your_model_name.hdf5')
new_model.summary()

new_model.compile(optimizer=new_model.optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

test_dir = os.path.join('your_test_image_path')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=1, target_size=(224, 224), color_mode='rgb',shuffle=False)
test_generator.reset()


output = new_model.predict_generator(test_generator, steps=1460) #number of total test imageset
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)

predicted_class_indices=np.argmax(output,axis=1)
labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

print(results)

class_names = ['af','normal','st','vf']  # Alphanumeric order1

print('-- Confusion Matrix --')
test_generator.reset()
Y_pred = new_model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

plot_confusion_matrix(confusion_matrix(test_generator.classes[test_generator.index_array], y_pred), normalize = True, target_names=class_names, title='Confusion matrix, without normalization')
plot_confusion_matrix(confusion_matrix(test_generator.classes[test_generator.index_array], y_pred), normalize = False, target_names=class_names, title='Confusion matrix, without normalization')

plt.show()
