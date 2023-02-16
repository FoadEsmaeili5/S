##########################
## Foad Esmaeili
## reference: https://www.amazon.com/Biomedical-Image-Analysis-Segmentation-Multimedia/dp/1598290207
## chapter 2
## section 2.1 -- 2.4
##########################

import numpy as np
import matplotlib.pyplot as plt
import skimage.filters as filt
import cv2

def snake(img,x,y,alpha = 0.001,beta = 0.4,sigma = 20,iteration = 1000,gamma = 10):
    N = np.size(x)
    # equation 2.19
    c =  gamma * (2 * alpha + 6*beta) +1
    b = gamma * (-alpha -4*beta)
    a = gamma * beta

    p = np.zeros((N,N),dtype = np.float)
    p[0] = np.c_[c,b,a,np.zeros((1,N-5)),a,b]
    for i in range(N):
        p[i]= np.roll(p[0],i) # 8.18
    
    p = np.linalg.inv(p)

    # computing Ix,Iy using gradient image
    smoothed = cv2.GaussianBlur((img-img.min()) / (img.max() - img.min()),(89,89),sigma) # making smooth image
    giy,gix = np.gradient(smoothed)
    gmi = (giy ** 2 + gix ** 2) ** 0.5
    gmi = (gmi - gmi.min())/(gmi.max() - gmi.min())
    
    Iy,Ix = np.gradient(gmi) # second derivative!

    def fmax(x,y):
        x[x<0] = 0
        y[y<0] = 0
        x[x>img.shape[1]-1] = img.shape[1]-1
        y[y>img.shape[0]-1] = img.shape[0]-1
        return y.round().astype(int),x.round().astype(int)
    
    for i in range(iteration):
        print(f"iteration{i}")
        fex = Ix[fmax(x,y)]
        fey = Iy[fmax(x,y)]
        
        x = np.dot(p,x + gamma * fex)
        y = np.dot(p,y + gamma * fey)
    return x,y



Image = cv2.imread('lip.jpg')
image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
img = np.array(image,dtype = np.float)


#define the initial snake
s = np.linspace(0, 2*np.pi, 400)
x = 210 + 210 * np.cos(s)
y = 90 + 75 * np.sin(s) 

#plot the image and results
plt.imshow(img,cmap = 'gray')
x_1,y_1= snake(img,x,y,iteration=200,gamma = 100)
plt.plot(x,y,'.')
plt.plot(x_1,y_1,'.')
plt.show()





#---------------------------------------------------------
from skimage.segmentation import active_contour
from skimage.filters import gaussian
img2 = gaussian(img,sigma = 20)
xx = active_contour(img2,np.c_[x,y],alpha = 0.01,beta = 0.04,gamma=1)
plt.imshow(img,cmap = "gray")
plt.plot(xx[:,0],xx[:,1],".")
plt.show()