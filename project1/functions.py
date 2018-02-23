import numpy as np

def scaleDown(pixel):
	output = np.zeros(len(pixel))
	for i in range(len(pixel)):
		output[i] = pixel[i] / 255.0
	return output

def scaleUp(pixel):
	for i in range(len(pixel)):
		pixel[i] = int(round(pixel[i] * 255))
	return pixel

def invGamma(pixel):
	for i in range(len(pixel)):
		if pixel[i] < 0.03928:
			pixel[i] = pixel[i] / 12.92
		else:
			pixel[i] = pow(((pixel[i] + 0.055) / 1.055), 2.4)
	return pixel

def gamma(pixel):
	for i in range(len(pixel)):
		if pixel[i] < 0.00304:
			pixel[i] = pixel[i] * 12.92
		else:
			pixel[i] = 1.055 * pow(pixel[i], 1 / 2.4) - 0.055
	return pixel

def RGB2XYZ(pixel):
	RGB = np.array([[pixel[2]], [pixel[1]], [pixel[0]]])
	convert = np.array([[0.412453, 0.35758, 0.180423],
		                [0.212671, 0.71516, 0.072169],
		                [0.019334, 0.119193, 0.950227]])
	XYZ = np.dot(convert, RGB)
	XYZ = np.array([XYZ[0][0], XYZ[1][0], XYZ[2][0]])
	return XYZ

def XYZ2RGB(pixel):
	XYZ = np.array([[pixel[0]], [pixel[1]], [pixel[2]]])
	convert = np.array([[3.240479, -1.53715, -0.498535],
		                [-0.969256, 1.875991, 0.041556],
		                [0.055648, -0.204043, 1.057311]])
	RGB = np.dot(convert, XYZ)
	RGB = np.array([RGB[0][0], RGB[1][0], RGB[2][0]])
	return RGB

def XYZ2Luv(pixel):
	X, Y, Z = pixel[0], pixel[1], pixel[2]
	L, u, v = 0.0, 0.0, 0.0
	d = X + 15 * Y + 3 * Z
	if d == 0:
		return np.array([L, u, v])

	if Y > 0.008856:
		L = 116 * pow(Y, 1.0 / 3) - 16
	else:
		L = 903.3 * Y
	u_prime = 4 * X / d
	v_prime = 9 * Y / d
	u = 13 * L * (u_prime - 0.19793943)
	v = 13 * L * (v_prime - 0.46831096)
	return np.array([L, u, v])

def Luv2XYZ(pixel):
	L, u, v = pixel[0], pixel[1], pixel[2]
	X, Y, Z = 0.0, 0.0, 0.0
	if L == 0:
		return np.array([X, Y, Z])
	u_prime = (u + 13 * 0.19793943 * L) / (13 * L)
	v_prime = (v + 13 * 0.46831096 * L) / (13 * L)
	if L > 7.999:
		Y = pow((L + 16) / 116, 3)
	else:
		Y = L / 903.3
	if v_prime == 0:
		X, Z = 0.0, 0.0
	else:
		X = Y * 2.25 * u_prime / v_prime
		Z = Y * (3 - 0.75 * u_prime - 5 * v_prime) / v_prime
	return np.array([X, Y, Z])

def limit(pixel):
	for i in range(len(pixel)):
		if pixel[i] < 0:
			pixel[i] = 0
		elif pixel[i] > 1:
			pixel[i] = 1
	return pixel 

