import cv2
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from evaluateCorrespondence import evaluateCorrespondence



def showCorrespondence(imageA, imageB, featuresA):
	fig = plt.figure()
	axA = fig.add_subplot(121)
	axB = fig.add_subplot(122)

	XA = []
	YA = []
	XB = []
	YB = []

	for item in featuresA:
		print('ITEM:')
		print item
		XA.append(item[0][0])#?????
		print(XA[0])
		YA.append(item[0][1])#?????
		print(YA[0])
		XB.append(item[2][1][0][0])
		print(XB[0])
		YB.append(item[2][1][0][1])
		print(YB[0])

	evaluateCorrespondence(XA, YA, XB, YB)



	i = 0

	while i < len(XA):

		axA.plot(XA,YA,'ko')
		axB.plot(XB,YB,'ko')

		xyA = (XA[i],YA[i])
		xyB = (XB[i],YB[i])

		con = ConnectionPatch(xyA, xyB, coordsA='data', coordsB='data', axesA=axB, axesB=axA, color='red')

		axB.add_artist(con)

		axA.plot(XA[i],YA[i],'ro',markersize=5)
		axB.plot(XB[i],YB[i],'ro',markersize=5)	

		#plt.show()


		i += 1


'''
def showCorrespondence(imageA, imageB, featuresA):
	fig = plt.figure()
	XA=[]
	YA=[]
	XB=[]
	YB = []

	for item in featuresA:
		print('ITEM:')
		print item
		XA.append(item[0][0])#?????
		print(XA[0])
		YA.append(item[0][1])#?????
		print(YA[0])
		XB.append(item[2][1][0][0])
		print(XB[0])
		YB.append(item[2][1][0][1])
		print(YB[0])

	#mngr = plt.get_current_fig_manager()
	# to put it into the upper left corner for example:
	#mngr.window.setGeometry(50,100,640, 545)
	plt.subplot(1,2,1)
	plt.imshow(imageA,cmap='gray')
	plt.subplot(1,2,2)
	plt.imshow(imageB,cmap='gray')

	for i in range(0,len(XA)):
		cur_color = np.random.rand(3,1).flatten()
		#print X1[i]
		plt.subplot(1,2,1)
		plt.plot(XA[i],YA[i], marker='o', ms=4, mec = 'k', mfc=cur_color,lw=2.0) #first
		# #cur_color = np.random.randint(255, size=3)
		plt.subplot(1,2,2);
		plt.plot(XB[i],YB[i], marker='o', ms=4, mec = 'k', mfc=cur_color,lw=2.0)
	
	fig.savefig('vis2.jpg')
	print 'Saving visualization to vis.jpg'
	
	return fig




'''
