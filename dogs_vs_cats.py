import os
import urllib.request
import zipfile
import numpy as np
from scipy import ndimage
import skimage.transform as sktrans
from sklearn import metrics
import matplotlib.pyplot as plt
import keras.backend as K

def image_files(train_file = "train.zip",train_folder = "train"):
    """ Download the training data from dropbox if it has not been downloaded and extract in train_folder folder."""
    if not os.path.exists(train_folder) and not os.path.exists(train_file):
        print("Proceding to download the data. This process may take some time..")
        urllib.request.urlretrieve("https://www.dropbox.com/s/8lbkqktfofzjraj/train.zip?raw=1", train_file)
        print("Done")
    else:
        print("data file {} has already been downloaded".format(train_file))

    # unzip files
    # If folder train does not exists extract all elements
    if not os.path.exists(train_folder):
        with zipfile.ZipFile(train_file, 'r') as myzip:
            myzip.extractall()
        print("Extracted")
    else:
        print("Data has already been extracted")

    return [os.path.join("train",img) for img in os.listdir("train")]

# mean = np.array([103.939,116.779,123.68])
def load_image_set(all_files, image_size=(3,50,50)):
    assert K.image_dim_ordering() == "th", "functions prepared to load data on NCHW format channel,height,width"
    """ load images into numpy arrays  order (channel,rows,cols). It substract load the images the mean"""
    feature_array = np.ndarray((len(all_files),)+image_size,dtype=np.float32)
    label = np.ndarray((len(all_files),),dtype=np.uint8)
    image_size_reord = (image_size[1],image_size[2],image_size[0])
    for i,image_path in enumerate(all_files):
        if i%100 == 0:
            print("loading image (%d/%d)"%(i+1,len(all_files)))
        image = ndimage.imread(image_path)        
        image_down = sktrans.resize(image, image_size_reord)
        #image_down -= mean
        feature_array[i] = image_down.transpose((2,0,1))
        label[i] = "dog" in image_path
    return feature_array, label
    
                  
def training_test_datasets(all_files,n_images_train=500,n_images_test=500,image_size=(3,50,50)):
    """
    Returns a randomly selected test and train datasets of the specified size.

    :param all_files: files to select train and test datasets
    :param n_images_train: number of training images
    :param n_images_test: number of test images
    :param image_size: 

    :return: train_features, train_labels, test_features, test_labels
    """
    files_selected = np.random.choice(np.array(all_files),n_images_train+n_images_test,replace=False)
    train_files = files_selected[:n_images_train]
    test_files = files_selected[n_images_train:]
    print("Loading train set")
    train_features, train_labels = load_image_set(train_files, image_size=image_size)

    print("Loading test set")
    test_features, test_labels = load_image_set(test_files, image_size=image_size)

    return train_features, train_labels,files_selected[:n_images_train], test_features, test_labels, files_selected[n_images_train:]

def plotROC(true_labels, test_probs_model):
    """
    plot ROC curve from true labels and probabilities.

    :param true_labels: true labels.
    :param test_probs_model: probabilities computed by a model (in sklearn usually the output of clf.predict_proba or clf.decision_function)
    :return:
    """
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, test_probs_model)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, 'k--',
             label='ROC (area = %0.2f)' % roc_auc, lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")





