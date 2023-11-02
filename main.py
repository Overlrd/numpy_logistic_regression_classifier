"""
Implementation from scratch of a simple Cat Vs Non Cat classifier using numpy

the steps we will follow are:
 - Initialize the parametres of the model
 - Learn the parameters for the model by minimizing the cost
 - Use the learned parameters to make predictions (on the test set)
 - Analyse the results and conclude
"""

import argparse
import pathlib
import sys
from PIL import Image
import numpy as np
import copy

from lr_utils import load_dataset, normalize_dataset, reshape_dataset, sigmoid, save_model, load_model
from public_tests import *


def initialize_with_zeros(dim):
    """
    creates a vector of shape (dim, 1) for w and initialize b to 0
    """
    w = np.zeros((dim, 1))
    b = 0.
    return w , b


def propagate(w, b, X, Y):
    """
    implements forward and backward propagation

        we compute A which is a vector or our model predictions 
    
        we compute the cost which is the discrepancy between our models prediction's A and the true labels Y
        computed by taking the mean of all individual losses for all samples in our dataset

        we compute dw which is the gradient of the loss in respect to the model parameter w.

        we compute dw which is the gradient of the loss in respect to b
    """
    m = X.shape[1]
    
    # forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum(((- np.log(A))*Y + (-np.log(1-A))*(1-Y)))/m
    #backward propagation
    dw = (np.dot(X, (A - Y).T))/m 
    db  = (np.sum(A - Y))/m

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}

    return grads,  cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimize w and b by running a gradient descent algorithm
    """
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    dw = None
    db = None

    costs = []

    for i in range(num_iterations):
        # compute gradient and cost
        grads, cost = propagate(w, b, X, Y)

        # retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update model parameters
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        # record the costs
        if i % 100 == 0:
            costs.append(cost)

            if print_cost:
                print(f"iteration {i}: cost = {cost}")

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # compute vector A predicting the probabilities of a cat being present in an image
    A = sigmoid(np.dot(w.T, X) + b)

    # vectorized implementation
    # in python you can perform arythmethics on booleans and integers
    # True == 1 (is true)
    Y_prediction = (A >= 0.5) * 1

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, classes=None, num_iterations=2000, learning_rate=0.5, print_cost=False, save = False, f_name = "model.pkl"):
    """
    build the logistic regression model by calling the functions above.
        
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zero
    w, b = initialize_with_zeros(X_train.shape[0])

    # run gradient descent
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # retrieve the learned weights
    w = params["w"]
    b = params["b"]

    # predicts the test set with the learned weights
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "classes": classes,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    if save:
        save_model(f_name, d)

    return d


def predict_custom_image(model_data_path, image_name, image_square_size):
    model_data = load_model(model_data_path)
    w = model_data["w"]
    b = model_data["b"]
    classes = model_data["classes"]

    fname = "images/" + image_name
    if not pathlib.Path(fname).exists():
        print(f"provided file {image_name} not found, file is not in the 'images' directory.")
        sys.exit()
    image = np.array(Image.open(fname).resize((image_square_size, image_square_size)))
    image = normalize_dataset(image)
    image = reshape_dataset(image, True, 1, image_square_size * image_square_size * 3)
    prediction =  predict(w, b, image)
    print("y = " + str(np.squeeze(prediction)) + ", your alogorithm predicts a \"" + classes[int(np.squeeze(prediction)),].decode("utf-8") + "\" picture." )

tests_completed = """
###########################################
#          All tests completed            #
###########################################
"""

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="train the logistic regression model", action="store_true")
parser.add_argument("-p", "--predict", help="predict whether a provided image contains a cat or not", action="store_true")
parser.add_argument("-f", "--file", help="the name of the image (in the images directory) to predict")
parser.add_argument("--tests", help="run tests on the differents components of the model", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.tests:
        sigmoid_test(sigmoid)
        propagate_test(propagate)
        optimize_test(optimize)
        predict_test(predict)
        model_test(model)       
        print(tests_completed)
    if args.train:
        train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(print_info=True)
        train_set_x = normalize_dataset(reshape_dataset(train_set_x_orig, True, train_set_x_orig.shape[0], -1))
        test_set_x = normalize_dataset(reshape_dataset(test_set_x_orig, True, test_set_x_orig.shape[0], -1))
        logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, classes, num_iterations=2000, learning_rate=0.005, print_cost=True, save=True)
    if args.predict:
        print('\033[93m')
        predict_custom_image("model.pkl", image_name=args.file, image_square_size=64)
        print('\033[0m')
