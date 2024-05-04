from PIL import Image
import numpy as np
from bm3d import bm3d, BM3DProfile
import matplotlib.pyplot as plt
import cv2 
PEAK = 0.005

def add_poisson_noise(x):
    x_new = np.zeros_like(x, dtype=np.float64)
    x_new = np.random.poisson(x  * PEAK) / PEAK
    return np.array(x_new, dtype=np.float64)
def blur(x):
    return cv2.GaussianBlur(x, (9,9), sigmaX=20, sigmaY=20)

def denoise(x,y,s,lam):
    k = 0
    u = (np.zeros_like(x, dtype=np.float64))
    v = (np.zeros_like(x, dtype=np.float64))  + 4.0*(np.sqrt(3.0/8.0)+1)
    while True:
        x1 = x
        x = ((lam*(v-u)-1)+np.sqrt((lam*(v-u)-1)**2+4*lam*y))/(2*lam)
        z = np.reshape(x+u, [s,s])
        v = bm3d(z, np.sqrt(100/lam))
        v = np.reshape(v, -1)
        u = u+(x-v)
        if k%10 == 0:
            print(np.linalg.norm(x1-x))
        if np.linalg.norm(x1-x) < 0.1:
            break
        
        if k > 30:
            break
        k = k+1
    return x

def deblur(x, y, s, lam):
    one = np.ones_like(x)
    k = 0
    u = (np.zeros_like(x, dtype=np.float64))
    v = (np.zeros_like(x, dtype=np.float64)) + 4.0*(np.sqrt(3.0/8.0)+1)
    while True:
        x1 = x
        k1 = 0
        while True:
            x2 = x
            temp = cv2.GaussianBlur(np.array(np.reshape(x,[s,s]), np.uint8), (9, 9), sigmaX=1.6, sigmaY=1.6)
            temp = np.reshape(temp, -1)
            temp = y/np.array(temp, np.float64)
            temp = cv2.GaussianBlur(np.array(np.reshape(temp,[s,s]),np.uint8), (9, 9), sigmaX=1.6, sigmaY=1.6)
            temp = np.array(np.reshape(temp, -1), np.uint8)
            temp1 = cv2.GaussianBlur(np.array(np.reshape(one,[s,s]),np.uint8), (9, 9), sigmaX=1.6, sigmaY=1.6)
            temp1 = np.array(np.reshape(temp1, -1), np.uint8)
            x = x- ((-temp)+temp1 + lam*(x-v+u))
            if np.linalg.norm(x2-x)<1:
                break
            k1 = k1 + 1
            if k1>100:
                break
        x = ((lam*(v-u)-1)+np.sqrt((lam*(v-u)-1)**2+4*lam*y))/(2*lam)
        z = np.reshape(x+u, [image1.shape[0], image1.shape[1]])
        v = bm3d(z, np.sqrt(100/lam))
        v = np.reshape(v, -1)
        u = u+(x-v)
        if k%10 == 0:
            print(np.linalg.norm(x1-x))
        if np.linalg.norm(x1-x) < 0.1:
            break
        #plt.imshow(np.reshape(x, [s,s]), cmap='grey')
        #plt.savefig(f"deblur/cameraman/cam_100_iter{k}.png")
        if k > 40:
            break
        k = k+1
    return x

image_name = "images/cameraman.png"
crop = 600 #Change accordingly
shp = crop
binning = False

if __name__  == "__main__":
    image = Image.open(image_name)
    image1 = np.asarray(image)
    if len(image1.shape)>2:
        image1 = image1[:crop, :crop,1]
    image1 = image1[:crop, :crop]
    plt.imshow(image1, cmap="gray")
    plt.show()
    plt.savefig(f"{image_name[:-4]}_crop.png")
    X = np.reshape(image1,-1)
    lam = 0.25
    y = blur(np.reshape(X, [image1.shape[0], image1.shape[1]]))
    y1 = np.reshape(y, -1)
    y2 = add_poisson_noise(y1)
    y3 = add_poisson_noise(X)

    y_noise = np.reshape(y3, [image1.shape[0], image1.shape[1]])
    y_blur = np.reshape(y2, [image1.shape[0], image1.shape[1]])

    #SHOW NOISY IMAGE
    plt.imshow(y_noise, cmap="gray")
    plt.show()
    plt.savefig(f"./denoise/noise_{image_name[:-4]}.png")

    #SHOW BLURRY IMAGE
    plt.imshow(y_blur, cmap="gray")
    plt.show()
    plt.savefig(f"./deblur/blurry_{image_name[:-4]}.png")

    if binning == True:
        y4 = cv2.resize(y_noise, [crop//3, crop//3])
        y3 = np.array(np.reshape(y4, -1), dtype = np.float64)
        shp = crop//3

    x_denoise = denoise(y3,y3,shp,lam*10)

    if binning == True:
        recon_1 = np.reshape(x_denoise, [crop//3, crop//3])   
        recon_1 = cv2.resize(recon_1, [crop,crop])
        x_denoise = np.array(np.reshape(recon_1, -1))
    recon_denoise = np.reshape(x_denoise, [crop,crop])

    plt.imshow(recon_denoise, cmap="gray")
    plt.show()
    plt.savefig(f"./denoise/recon_{image_name[:-4]}_denoise.png")

    x_deblur = deblur(y2, y2, image1.shape[0],lam)
    recon_deblur = np.reshape(x_deblur, [image1.shape[0], image1.shape[1]])
    plt.imshow(recon_deblur, cmap="gray")
    plt.show()
    plt.savefig(f"./deblur/recon_{image_name[:-4]}_deblur.png")