
# Local Feature Stencil Code
# CS 589 Computater Vision, American Uniersity, Bei Xiao
# Adapted from James Hayes's MATLAB starter code for Project.2

# % This script 
	# % (1) Loads and resizes images
	# % (2) Finds interest points in those images                 (you code this)
	# % (3) Describes each interest point with a local feature    (you code this)
	# % (4) Finds matching features                               (you code this)
	# % (5) Visualizes the matches
	# % (6) Evaluates the matches based on ground truth correspondences

	# % There are numerous other image sets in the data sets folder uploaded. 
	# % You can simply download images off the Internet, as well. However, the
	# % evaluation function at the bottom of this script will only work for this
	# % particular image pair (unless you add ground truth annotations for other
	# % image pairs). It is suggested that you only work with these two images
	# % until you are satisfied with your implementation and ready to test on
	# % additional images. 

	# A single scale pipeline works fine for these two
	# images (and will give you full credit for this project), but you will
	# need local features at multiple scales to handle harder cases.


	# % You don't have to work with grayscale images. Matching with color
	# % information might be helpful.

print('Start proj4.py')

import cv2
import numpy as np
from scipy import misc
import scipy.io
from matplotlib import pyplot as plt
from getInterestPoints import getInterestPoints
from getFeatures import getFeatures
from matchFeatures import matchFeatures
from showCorrespondence import showCorrespondence 
from evaluateCorrespondence import evaluateCorrespondence

print('proj4.py: imports complete')

# read in the notre dame images
imageA = cv2.imread('../data/Notre Dame/921919841_a30df938f2_o.jpg') #'Office/Left.jpg
imageB = cv2.imread('../data/Notre Dame/4191453057_c86028ce1f_o.jpg') #')Office/Center.jpg
print('proj4.py: Images read')


# convert to grayscale
imageA =  cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
imageB =  cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

#scale_factor = 0.5; #make images smaller to speed up the algorithm #THIS IS COMMENTED OUT BECAUSE IT DOESN"T DO ANYTHING

# These 4 lines cut down both images to half size
heightA, widthA = imageA.shape[:2]
heightB, widthB = imageB.shape[:2]
imageA = cv2.resize(imageA,(widthA/2, heightA/2), interpolation = cv2.INTER_CUBIC)
imageB = cv2.resize(imageB,(widthB/2,heightB/2), interpolation = cv2.INTER_CUBIC)  #height1

feature_width = 16; #width and height of each local feature, in pixels. 

print('proj4.py: images grey-shifted and scaled down to half size')

# %% Find distinctive points in each image. Szeliski 4.1.1
# % !!! You will need to implement get_interest_points. !!!
imageAPoints = getInterestPoints(imageA, feature_width) #this should return a list of 2 value tuples
imageBPoints = getInterestPoints(imageB, feature_width) 

'''
print('proj4.py: got interest points')
print(imageAPoints)
'''

# %% Create feature vectors at each interest point. Szeliski 4.1.2
# % !!! You will need to implement get_features. !!!

# intake a set of points. Output a set of features
imageA_features = getFeatures(imageA, imageAPoints, feature_width)

imageB_features = getFeatures(imageB, imageBPoints, feature_width)

matched = matchFeatures(imageA_features, imageB_features)

print('proj4.py: matched')
#print(matched)

showCorrespondence(imageA, imageB, matched)


#turn feature list into 2 lists of coords
#for feat in matched:
	#coordAlist.append(feat[0])
	#coordBlist.append(feat[2][1][0]) ??????


'''
# %% Match features. Szeliski 4.1.3
# % !!! You will need to implement get_features. !!!
[matches, confidences] = matchFeatures(imageA_features, imageB_features)

# % You might want to set 'num_pts_to_visualize' and 'num_pts_to_evaluate' to
# % some constant once you start detecting hundreds of interest points,
# % otherwise things might get too cluttered. You could also threshold based
# % on confidence.
num_pts_to_visualize = matches.shape[0]

show_correspondence(imageA, imageB, x1[matches[0:num_pts_to_visualize,0:1]],
	y1[matches[0:num_pts_to_visualize,0:1]],
	x2[matches[0:num_pts_to_visualize,1:2]],
	y2[matches[0:num_pts_to_visualize,1:2]])

num_pts_to_evaluate = matches.shape[0]

# you can also end your code by this:

#fig.savefig('vis.jpg')
#print 'Saving visualization to vis.jpg'

# # % All of the coordinates are being divided by scale_factor because of the
# # % imresize operation at the top of this script. This evaluation function
# # % will only work for the particular Notre Dame image pair specified in the
# # % starter code. You can simply comment out
# # % this function once you start testing on additional image pairs.

evaluateCorrespondence(x1[matches[0:num_pts_to_evaluate,0:1]]/scale_factor,
                        y1[matches[0:num_pts_to_evaluate,0:1]]/scale_factor,
                        x2[matches[0:num_pts_to_evaluate,1:2]]/scale_factor,
                        y2[matches[0:num_pts_to_evaluate,1:2]]/scale_factor)





'''



