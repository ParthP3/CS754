from PIL import Image
import numpy as np
from bm3d import bm3d, BM3DProfile
import matplotlib.pyplot as plt
import cv2 

def add_poisson_noise(x,lam):
    return x + np.random.poisson(lam, X.shape)
def blur(x):
    return cv2.GaussianBlur(x, (9,9), sigmaX=100, sigmaY=100)

def denoise(x,lam):
    k = 0
    u = (np.zeros_like(x, dtype=np.float64))
    v = (np.zeros_like(x, dtype=np.float64))
    while True:
        x1 = x
        x = ((lam*(v-u)-1)+np.sqrt((lam*(v-u)-1)**2+4*lam*y))/(2*lam)
        z = np.reshape(x+u, [image1.shape[0], image1.shape[1]])
        v = bm3d(z, np.sqrt(1/lam))
        v = np.reshape(v, -1)
        u = u+(x-v)
        if k%10 == 0:
            print(np.linalg.norm(x1-x))
        if np.linalg.norm(x1-x) < 0.1:
            break
        if k > 50:
            break
        k = k+1
    return x

def deblur(x, s, lam):
    one = np.ones_like(x)
    k = 0
    u = (np.zeros_like(x, dtype=np.float64))
    v = (np.zeros_like(x, dtype=np.float64))
    while True:
        x1 = x
        k1 = 0
        while True:
            x2 = x
            temp = cv2.GaussianBlur(np.reshape(x,[s,s]), (9, 9), sigmaX=10, sigmaY=10)
            temp = np.reshape(temp, -1)
            temp = y/temp
            temp = cv2.GaussianBlur(np.reshape(temp,[s,s]), (9, 9), sigmaX=10, sigmaY=10)
            temp = np.reshape(temp, -1)
            temp1 = cv2.GaussianBlur(np.reshape(one,[s,s]), (9, 9), sigmaX=10, sigmaY=10)
            temp1 = np.reshape(temp1, -1)
            x = x- ((-temp)+temp1 + lam*(x-v+u))
            if np.linalg.norm(x2-x)<1:
                break
            k1 = k1 + 1
            if k1>100:
                break
        x = ((lam*(v-u)-1)+np.sqrt((lam*(v-u)-1)**2+4*lam*y))/(2*lam)
        z = np.reshape(x+u, [image1.shape[0], image1.shape[1]])
        v = bm3d(z, np.sqrt(1/lam))
        v = np.reshape(v, -1)
        u = u+(x-v)
        if k%10 == 0:
            print(np.linalg.norm(x1-x))
        if np.linalg.norm(x1-x) < 0.1:
            break
        if k > 50:
            break
        k = k+1
    return x
#LOAD IMAGE
image = Image.open("barbara.jpg")
image1 = np.asarray(image)
image1 = image1[0:256, 0:256]
plt.imshow(image1, cmap="gray")
plt.show()
plt.savefig("barb_crop.png")
X = np.reshape(image1,-1)
print(X.shape)

lam = 100
y = blur(np.reshape(X, [image1.shape[0], image1.shape[1]]))
y = np.reshape(y, -1)
y = add_poisson_noise(y,lam)

y1 = np.reshape(y, [image1.shape[0], image1.shape[1]])
#SHOW NOISY IMAGE
plt.imshow(y1, cmap="gray")
plt.show()
plt.savefig("noise_barb.png")


#x_denoise = denoise(y,lam)
x_deblur = deblur(y,image1.shape[0],lam)
#recon_denoise = np.reshape(x_denoise, [image1.shape[0], image1.shape[1]])
recon_deblur = np.reshape(x_deblur, [image1.shape[0], image1.shape[1]])

#plt.imshow(recon_denoise, cmap="gray")
#plt.show()
#plt.savefig("recon_denoise.png")
plt.imshow(recon_deblur, cmap="gray")
plt.show()
plt.savefig("recon_denoise.png")