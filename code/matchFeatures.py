# % Local Feature Stencil Code
# CSC 589 Intro to computer vision. 
# Code adapated from MATLAB written by James Hayes from Brown University.

# % 'features1' and 'features2' are the n x feature dimensionality features
# %   from the two images.
# % If you want to include geometric verification in this stage, you can add
# % the x and y locations of the features as additional inputs.
# %
# % 'matches' is a k x 2 matrix, where k is the number of matches. The first
# %   column is an index in features 1, the second column is an index
# %   in features2. 
# % 'Confidences' is a k x 1 matrix with a real valued confidence for every
# %   match.
# % 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.

import cv2
import numpy as np

def matchFeatures(featuresA, featuresB):

	# % This function does not need to be symmetric (e.g. it can produce
	# % different numbers of matches depending on the order of the arguments).

	# % To start with, simply implement the "ratio test", equation 4.18 in
	# % section 4.1.3 of Szeliski. For extra credit you can implement various
	# % forms of spatial verification of matches.

	# % Placeholder that you can delete. Random matches and confidences

	# FEATURE MUST BE IN FORMAT [[x,y],[HISTOGRAM]]
	num_features1 = len(featuresA)
	num_features2 = len(featuresB)

	for aFeat in featuresA:
		aCoord = aFeat[0]
		aHisto = aFeat[1]
		aFeat.append([1000000,100]) #sets default distance at an arbitrarily high value so the distance check feature match catches
		#find the one that is most similar, then attach it to feature A
		for bFeat in featuresB:
			bCoord = bFeat[0]
			bHisto = bFeat[1]

			dist = 0
			for pos, val in enumerate(aHisto): #?While?
				dist += abs(val - bHisto[pos])

				#subtract that value in each histo from eachother (absolute value)
				#then add value to total distance thing
			if dist < aFeat[2][0]:		#aFeat = [[x,y],[HISTOGRAM],[dist,bFeat]]
				aFeat[2] = [dist, bFeat]


			#print('matchFeatures.py aCoord: ' + str(aCoord))
			#print('matchFeatures.py bCoord: ' + str(bCoord))

		print(str(aCoord) + ' --- ' + str(aFeat[2][1][0]))
	# this is annoying for Python, if you want the number to be integer, you must specify its data type


	return featuresA


