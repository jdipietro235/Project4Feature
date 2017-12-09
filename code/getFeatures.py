import cv2
import numpy as np
from scipy import signal

# % Local Feature Stencil Code
# % Returns a set of feature descriptors for a given set of interest points. 

	# % 'image' can be grayscale or color, your choice.
	# % 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
	# %   The local features should be centered at x and y.
	# % 'feature_width', in pixels, is the local feature width. You can assume
	# %   that feature_width will be a multiple of 4 (i.e. every cell of your
	# %   local SIFT-like feature will have an integer width and height).
	# % If you want to detect and describe features at multiple scales or
	# % particular orientations you can add input arguments.

	# % 'features' is the array of computed features. It should have the
	# %   following size: [length(x) x feature dimensionality] (e.g. 128 for
	# %   standard SIFT)

def getFeatures(image, interestPoints, feature_width):

# % To start with, you might want to simply use normalized patches as your
	# % local feature. This is very simple to code and works OK. However, to get
	# % full credit you will need to implement the more effective SIFT descriptor
	# % (See Szeliski 4.1.2 or the original publications at
	# % http://www.cs.ubc.ca/~lowe/keypoints/)

	# % Your implementation does not need to exactly match the SIFT reference.
	# % Here are the key properties your (baseline) descriptor should have:
	# %  (1) a 4x4 grid of cells, each feature_width/4.
	# %  (2) each cell should have a histogram of the local distribution of
	# %    gradients in 8 orientations. Appending these histograms together will
	# %    give you 4x4 x 8 = 128 dimensions.
	# %  (3) Each feature should be normalized to unit length
	# %
	# % You do not need to perform the interpolation in which each gradient
	# % measurement contributes to multiple orientation bins in multiple cells
	# % As described in Szeliski, a single gradient measurement creates a
	# % weighted contribution to the 4 nearest cells and the 2 nearest
	# % orientation bins within each cell, for 8 total contributions. This type
	# % of interpolation probably will help, though.

	# % You do not have to explicitly compute the gradient orientation at each
	# % pixel (although you are free to do so). You can instead filter with
	# % oriented filters (e.g. a filter that responds to edges with a specific
	# % orientation). All of your SIFT-like feature can be constructed entirely
	# % from filtering fairly quickly in this way. 

	# For filtering the image and computing the gradients, 
	# you can either use the following functions or implement you own filtering code as you did in the second project:
	#scipy.ndimage.sobel: Filters the input image with Sobel filter.
	#scipy.ndimage.gaussian_filter: Filters the input image with a Gaussian filter.
	#scipy.ndimage.filters.maximum_filter: Filters the input image with a maximum filter.
	#scipy.ndimage.filters.convolve: Filters the input image with the selected filter.

	# % You do not need to do the normalize -> threshold -> normalize again
	# % operation as detailed in Szeliski and the SIFT paper. It can help, though.

	# % Another simple trick which can help is to raise each element of the final
# % feature vector to some power that is less than one.

# % Placeholder that you can delete. Empty features.
   	#features = np.zeros((len(x), 128));
	#return features 

	featureList = []

	for pointCount, point in enumerate(interestPoints):
		if point[0] - 8 < 1 or point[1] - 8 < 1 or point[0] + 8 > image.shape[0] or point[1] + 8 > image.shape[1]:
			print('getFeatures.py: passed')
			continue
		else:
			newPatch = makePatch(point, image)
			newHisto = makeHisto(newPatch)
			featureList.append([point,newHisto])
	return featureList

def makePatch(point, image):
	
	i = -8
	patch = []
	while i < 8:
		newVertList = []
		j = 8
		
		while j > -8:
			try:
				newVertList.append(image[point[0] + i,point[1] + j])
			except:
				print('getFeatures.py: Exception')
			j -= 1
		patch.append(newVertList)
		i += 1
	'''
	print('getFeatures.py: Patch length:' + str(len(patch)))
	print('getFeatures.py: Patch width:' + str(len(patch[0])))
	'''
	'''
	for layer in patch:
		print(layer)
	'''

	return patch

	
def makeHisto(patch):

	# now plit into 4 parts
	patch = np.array(patch)

	h, w = patch.shape
	outPatch = (patch.reshape(h//4, 4, -1, 4).swapaxes(1,2).reshape(-1, 4, 4))

	'''
	print('outPatch:')
	print(outPatch)
	print(outPatch[0])
	'''

	#now calculate grandient magnitude and orientation for each box

	#histo = [[],[],[],[],[],[],[],[]]
	histo = [0,0,0,0,0,0,0,0]

	for box in outPatch:

		sobelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])   
		sobelY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

		#boxDx = signal.convolve(box, sobelX, mode='same') # horizontal derivative   
		#boxDy = signal.convolve(box, sobelY, mode='same')  # vertical derivative

		boxDx = cv2.Sobel(box, cv2.CV_32F, 1, 0, ksize=1)
		boxDy = cv2.Sobel(box, cv2.CV_32F, 0, 1, ksize=1)

		boxMag, boxOri = cv2.cartToPolar(boxDx, boxDy, angleInDegrees=True)

		'''
		boxGrad = boxDx + boxDy
		boxMag = cv2.magnitude(boxDx, boxDy)
		boxOri = cv2.phase(sobelX, sobelY, True)
		'''
		'''
		print('Box orientation: ')
		print(boxOri)
		'''

		for row in boxOri:
			for pixel in row:
				#print('pixel bin:')
				#print(pixel)
				yourBin = (int(pixel / 45))
				histo[yourBin] += 1

	#print(histo)
	return histo
		

		# one histogram for the entire grid
		# compare every feature against every feature
		# find orientation via math
		# bin it?

	'''
		ddepth = cv2.CV_32F
		dx = cv2.Sobel(box, ddepth, 1, 0)
		dy = cv2.Sobel(box, ddepth, 0, 1)
		dxabs = cv2.convertScaleAbs(dx)
		dyabs = cv2.convertScaleAbs(dy)
		mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
		print('MAG: ' + str(mag))
		'''



'''
> Step 1
	Pick a point. Just go down the list of points
> Step 2
	Put a window around that point
	Window should be the size of feature size
> Step 3
	Split the window into 4*4 grid
> Step 4
	Calculate gradient magnitudes and orientations in each grid
> Step 5



'''








