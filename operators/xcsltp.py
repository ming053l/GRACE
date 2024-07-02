""" Extended Center Symmetric Local Ternary Pattern """

from tqdm import tqdm
import numpy as np

def heaviside(x, y):
	
	if x > 3 and y > 3:
		return 2
	elif x < -3 and y < -3:
		return 1
	else:
		return 0

def get_features(img):
    
    img_height, img_width = img.shape
    zeroHorizontal = np.zeros(img_width + 2).reshape(1, img_width + 2)
    zeroVertical = np.zeros(img_height).reshape(img_height, 1)


    img = np.concatenate((img, zeroVertical), axis = 1)
    img = np.concatenate((zeroVertical, img), axis = 1)
    img = np.concatenate((zeroHorizontal, img), axis = 0)
    img = np.concatenate((img, zeroHorizontal), axis = 0)

    pattern_img = np.zeros((img_height + 1, img_width + 1))

    for x in range(1, img_height + 1):
        for y in range(1, img_width + 1): 

            s1 = heaviside(img[x-1, y-1]-img[x, y], img[x, y]-img[x+1, y+1]) 
            s3 = heaviside(img[x-1, y+1]-img[x, y], img[x, y]-img[x+1, y-1])*3

            s = s1 + s3

            pattern_img[x, y] = s

    pattern_img = pattern_img[1:(img_height+1), 1:(img_width+1)].astype(int) 		
#     histogram = np.histogram(pattern_img, bins = np.arange(17))[0]
#     histogram = histogram.reshape(1, -1)
    


    return pattern_img