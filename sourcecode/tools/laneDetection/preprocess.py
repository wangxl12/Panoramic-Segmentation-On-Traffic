import cv2
import numpy as np

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def warp(image):
    w = image.shape[1]
    h = image.shape[0]
    A = [w*0.1625, h]
    B = [w*0.898, h]
    C = [w*0.3406, 0.56*h]
    D = [w*0.71, 0.56*h]
    # A = [w * 0.26, h]
    # B = [w * 0.81, h]
    # C = [w * 0.35, h * 0.45]
    # D = [w * 0.68, h * 0.45]
    # 原图像的大小是（720，1280,3）

    # src = np.float32([[200, 460], [1150, 460], [436, 220], [913, 220]])
    # dst = np.float32([[300, 720], [1000, 720], [400, 0], [1200, 0]])
    src = np.float32([A, B, C, D])
    dst = np.float32([[w*0.2344,h],[w*0.78125,h], [w*0.3125, C[1]*0], [w*0.9375, D[1]*0]])
    # print(src)
    # print(dst)
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, invM

def threshold(image):
    ret, image = cv2.threshold(image, 220, 225, cv2.THRESH_BINARY)
    if(ret == False):
        print('Error in thresholding')
    else:
        return image