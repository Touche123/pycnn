#import numpy as np
#from numpy import asarray
import cv2
import matplotlib.pyplot as plt
import math


#Logistic (aka soft step). Range 0 to 1. Centered at 0.5
#Suffers from vanishing gradient problem.
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#Hyperbolic tangent function - Tanh. Range -1 to 1. Zero centered
#Suffers from vanishing gradient problem
def TanH(x):
    return (2 / (1 + math.exp(-2*x))) - 1

#Rectified linear
#Recommended for MLP and CNNs, but probobly not RNNs
#Start with low bias values, such as 0.1. (might not be true)
#Use HE weight initialization
#Scale input data. Either standardize variables to have a zero mean and unit variance or normalize each value to the scale 0 to 1
#In some cases it may be a good idea to use a form of weight regularization, such as an L1 or L2 vector norm
def ReLu(x):
    return max(0.0, x)

#Leaky ReLu
def LReLu(x):
    pass

#Exponential linear unit
def ELU(x):
    pass

#Parametric ReLu
def PReLU(x):
    pass

#Piecewise linear function that returns the maximum of the inputs
#designed to be used in conjunction with the dropout regularization technique.
def MaxOut(x):
    pass

def main():
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE) / 255
    plt.imshow(img, cmap = 'gray')
    plt.show()

if __name__ == "__main__":
    main()

class Conv_Op:

    def __init__(self, num_filters, filter_size):
        pass

    def image_region(self, image):
        pass
    def forward_prop(self, image):
        pass