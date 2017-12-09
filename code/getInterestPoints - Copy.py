
print('Start getInterestPoints.py')

import cv2
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt

print('getInterestPoints.py: imports complete')

# % Local Feature Stencil Code
# CSC 589 Intro to computer vision. 
# Code adapated from MATLAB written by James Hayes from Brown University
# % Returns a set of interest points for the input image

# % 'image' can be grayscale or color, your choice.
# % 'feature_width', in pixels, is the local feature width. It might be
# %   useful in this function in order to (a) suppress boundary interest
# %   points (where a feature wouldn't fit entirely in the image, anyway)
# %   or(b) scale the image filters being used. Or you can ignore it.

# % 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
# % 'confidence' is an nx1 vector indicating the strength of the interest
# %   point. You might use this later or not.
# % 'scale' and 'orientation' are nx1 vectors indicating the scale and
# %   orientation of each interest point. These are OPTIONAL. By default you
# %   do not need to make scale and orientation invariant local features.

def getInterestPoints(image, feature_width):
    
# % Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
# % You can create additional interest point detector functions (e.g. MSER)
# % for extra credit.

# % If you're finding spurious interest point detections near the boundaries,
# % it is safe to simply suppress the gradients / corners near the edges of
# % the image.

# % The lecture slides and textbook are a bit vague on how to do the
# % non-maximum suppression once you've thresholded the cornerness score.
# % You are free to experiment. The Solem textbook provided some nice code for this. 

    # Placeholder that you can delete. 20 random points.  
    '''
    rowCount,colCount = image.shape
    x = np.ceil(np.random.rand(20,1) * rowCount)
    y = np.ceil(np.random.rand(20,1) * colCount)
    '''

    harrisIm = computeHarrisResponse(image)
    filteredCoords = getHarrisPoints(harrisIm,12,0.1)
    print(filteredCoords)
    plotHarrisPoints(harrisIm,filteredCoords)

    #confidence = np.random.rand(len(x),1)
    #scale = np.random.rand(len(x),1)
    #orientation = np.random.rand(len(x),1)

    #return many points
    return filteredCoords # it is optional to return scale and orientation

"""
def harrisDetection():
    # compute x and y derivitaves of image
        # 
    # compute products of deri

    gradY, gradX = np.gradient(img)
"""

def getHarrisPoints(harrisim,min_dist = 0, threshold = 0.1): # threshold=0.1
    # find top corner candiates above a threshold

    print('Image Maximum: ')
    print(harrisim.max())

    corner_threshold = harrisim.max() * threshold

    print('corner_threshold')
    print(corner_threshold)

    harrisim_t = harrisim > corner_threshold
    
    #get the coordinates, all the non-zero components 
    coords = np.array(harrisim_t.nonzero()).T

    print('Coords:')
    print(coords)
    
    # ...add their values
    candidateVals = [harrisim[c[0],c[1]] for c in coords]
    
    # sort candidates in descending order of corner responses
    index = np.argsort(candidateVals)
    
    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    
    # select the best points taking min_dist into account
    filtered_coords = [] 
    print('Allowed locations:')
    print(allowed_locations)
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0     
    return filtered_coords
    

def computeHarrisResponse(image):
    """ compute the Harris corner detector response function 
        for each pixel in the image"""

    #derivatives
    imXderiv,imYderiv = gaussDerivatives(image, 3)

    #kernel for blurring
    gaussKern = matlabGauss2D((3,3),3)

    #compute components of the structure tensor
    xxComponent = signal.convolve(imXderiv*imXderiv, gaussKern, mode='same')
    xyComponent = signal.convolve(imXderiv*imYderiv, gaussKern, mode='same')
    yyComponent = signal.convolve(imXderiv*imYderiv, gaussKern, mode='same')

    #determinant and trace
    determinant = xxComponent * yyComponent - xyComponent**2
    trace = xxComponent + yyComponent

    return determinant / trace


def plotHarrisPoints(image,filtered_coords):
    plt.figure()
    plt.imshow(image,cmap='gray')
    plt.plot([p[1]for p in filtered_coords], [p[0] for p in filtered_coords],'*')
    plt.show()


# write a 2D gaussian kernal
def matlabGauss2D(shape=(3,3),sigma=1):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x*x + y*y)/(2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
    
# construct derivative kernals from scratch
def gaussDerivativeKernels(size, sizey=None):
    """ returns x and y derivatives of a 2D 
        gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]

    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)

    #
    #
    #*****************************************************8
    gx = - x * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    gy = - y * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 

    return gx,gy    
    
# computing derivatives     
def gaussDerivatives(im, n, ny=None):
    """ returns x and y derivatives of an image using gaussian 
        derivative filters of size n. The optional argument 
        ny allows for a different size in the y direction."""

    gx,gy = gaussDerivativeKernels(n, sizey=ny)

    imx = signal.convolve(im,gx, mode='same')
    imy = signal.convolve(im,gy, mode='same')

    return imx,imy
    
#gx,gy = gaussDerivativeKernels(3)


    



