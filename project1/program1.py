import sys
import cv2
import numpy as np
import os
sys.path.append(os.path.abspath("."))
from functions import * 	

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)

Lmin, Lmax = 255.0, 0.0
for i in range(H1, H2):
    for j in range(W1, W2):
    	pixel = XYZ2Luv(RGB2XYZ(invGamma(limit(scaleDown(inputImage[i, j])))))
    	L, u, v = pixel[0], pixel[1], pixel[2]
    	Lmax = max(Lmax, L)
    	Lmin = min(Lmin, L)

for i in range(0, rows):
	for j in range(0, cols):
		pixel = XYZ2Luv(RGB2XYZ(invGamma(limit(scaleDown(inputImage[i, j])))))
		L, u, v = pixel[0], pixel[1], pixel[2]
		if L > Lmax:
			L = 100
		elif L < Lmin:
			L = 0
		newPixel = np.array([L, u, v])
		r, g, b = scaleUp(limit(gamma(XYZ2RGB(Luv2XYZ(newPixel)))))
		outputImage[i, j] = b, g, r

for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
    	if i == H1 or i == H2 or j == W1 or j == W2:
    		inputImage[i, j] = [0, 0, 0]
    		
cv2.imshow("input image: " + name_input, inputImage)
cv2.imshow("output:", outputImage)
cv2.imwrite(name_output, outputImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
