import sys
import pickle
import numpy as np
import h5py
   
def sigmoid(z):
    """
    We use sigmoid to rescale our ŷ prediction between 0 and 1 as we are doing binary classification.
    and we are looking for ŷ to be the probability y is 1.
    the sigmoid formula is sigmoid(𝑧)=1/1+𝑒^(−𝑧) for 𝑧=𝑤𝑇𝑥+𝑏
    """
    s = 1 / (1 + np.exp(-z))
    return s
    
    
def load_dataset(print_info = False):
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    if print_info:
        m_train = train_set_x_orig.shape[0]
        m_test = test_set_x_orig.shape[0]
        num_px = train_set_x_orig.shape[1]

        print ("Number of training examples: m_train = " + str(m_train))
        print ("Number of testing examples: m_test = " + str(m_test))
        print ("Height/Width of each image: num_px = " + str(num_px))
        print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print ("train_set_x shape: " + str(train_set_x_orig.shape))
        print ("train_set_y shape: " + str(train_set_y_orig.shape))
        print ("test_set_x shape: " + str(test_set_x_orig.shape))
        print ("test_set_y shape: " + str(test_set_y_orig.shape))


    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def normalize_dataset(dataset):
    return dataset / 255.

def reshape_dataset(dataset, transpose = True, *args):
    dataset = dataset.reshape(*args)
    if transpose:
        dataset = dataset.T
    return dataset

def save_model(fname, model_data):
    with open(fname, 'wb') as file:
        pickle.dump(model_data, file)

def load_model(fname):
    try:
        with open(fname, 'rb') as file:
            loaded_model_data = pickle.load(file)
        return loaded_model_data
    except FileNotFoundError:
        # Handle the case when the file is not found
        print(f"\033[91m{fname} don't match any file, saved model file not found\033[0m")  # Print in red and reset color
        sys.exit()
