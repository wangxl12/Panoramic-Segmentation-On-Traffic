from time import time 
import cv2
import numpy as np 
import copy 
from tqdm import tqdm

def left_line_get_x(top, bottom, h):
    y = h
    x = (y - (top[1] - (top[1] - bottom[1]) * top[0]/ (top[0] - bottom[0]))) * (top[0] - bottom[0]) / (top[1] - bottom[1])
    return (int(x), y)

def right_line_get_x(top, bottom, h):
    y = h
    x = (y - (top[1] - (top[1] - bottom[1]) * top[0]/ (top[0] - bottom[0]))) * (top[0] - bottom[0]) / (top[1] - bottom[1])
    return (int(x), y)

def roi_mask(img, vertices):
    mask = np.zeros_like(img)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def get_car_lane(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    # blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    canny_img = cv2.Canny(gray, 80, 80)
    # 梯形
    roi_vtx = np.array([[(w * 0.26, h * 0.65), (w * 0.16, h), (w * 0.84, h), (w * 0.74, h * 0.65)]], dtype='int32')
#     roi_vtx = np.array([[(w * 0.2, h * 0.65), (0, 0.70 * h), (0, h), (w, h), (w, 0.70 * h), (w * 0.80, h * 0.65)]], dtype='int32')
    # print(roi_vtx)
    roi_edges = roi_mask(canny_img, roi_vtx)
    minLineLength = 50  # 组成一条直线最少点的数量
    maxLineGap = 10  # 一条直线上的亮点的最大距离
    threshold = 10
    # cv2.imshow('fff', roi_edges)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(roi_edges, 0.1, np.pi/180, threshold, minLineLength, maxLineGap)
    if lines is None:
        return img
    # 筛选掉常见的不符合的线：
    # 去掉水平和垂直
    lines = [line[0] for line in lines if not (abs(line[0][0] - line[0][2]) <= 10 or abs(line[0][1] - line[0][3]) <= 10)]
    # 去掉水平夹角小于45度的
    lines = [line for line in lines if abs((line[1] - line[3]) / (line[0] - line[2])) > (np.pi / 4.)]
    # 将左边的线和右边的线分开,并筛选掉垂直和水平的线段
    right_lines = [line for line in lines if line[3] - line[1] > 0 and line[2] - line[0] > 0]
    left_lines = [line for line in lines if not (line[3] - line[1] > 0 and line[2] - line[0] > 0)]
    # 从左右线的集合中选出符合的线
    if left_lines == [] or right_lines == []:
        return img
    left_bottom = min(left_lines, key=lambda x: min(x[0], x[2]))
    left_top = max(left_lines, key=lambda x: max(x[0], x[2]))
    right_bottom = max(right_lines, key=lambda x: max(x[0], x[2]))
    right_top = min(right_lines, key=lambda x: min(x[0], x[2]))
    # 将线的四个坐标点配对并选出四个角的坐标
    left_bottom = [(left_bottom[0], left_bottom[1]), (left_bottom[2], left_bottom[3])]
    left_top = [(left_top[0], left_top[1]), (left_top[2], left_top[3])]
    right_bottom = [(right_bottom[0], right_bottom[1]), (right_bottom[2], right_bottom[3])]
    right_top = [(right_top[0], right_top[1]), (right_top[2], right_top[3])]
    left_bottom = min(left_bottom, key=lambda x: x[0])
    left_top = max(left_top, key=lambda x: x[0])
    right_bottom = max(right_bottom, key=lambda x: x[0])
    right_top = min(right_top, key=lambda x: x[0])
    # print(left_top, left_bottom, right_bottom, right_top)
    if not ((right_top[0] - right_bottom[0]) <= 1e-4 or (right_top[0] - right_bottom[0]) <= 1e-4):
        # 将车道线的上边缘整理成水平
        left_right_function = {'left': left_line_get_x, 'right': right_line_get_x}
        if left_top[1] < right_top[1]:
            right_top = left_right_function['right'](right_top, right_bottom, left_top[1])
        elif left_top[1] > right_top[1]:
            left_top = left_right_function['left'](left_top, left_bottom, right_top[1])
        # 将四个坐标扩展一下(扩展到与图片的下边缘重合)
        left_bottom = left_right_function['left'](left_top, left_bottom, h)
        right_bottom = left_right_function['right'](right_top, right_bottom, h)

    img_copy = copy.deepcopy(img)
    # print(left_bottom, right_bottom)
    cv2.fillPoly(img_copy, np.array([[left_bottom, left_top, right_top, right_bottom]]), (255, 0, 0))
    # cv2.line(img, left_top, left_bottom, (0, 255, 0), 2)
    # cv2.line(img, right_top, right_bottom, (0, 0, 255), 2)
    img_add = cv2.addWeighted(img, 0.4, img_copy, 0.6, 0)
    # for line in lines:
    #     x1,y1,x2,y2 = line[0]
    #     print(line)
    #     if x1 == x2 or y1 == y2:
    #         continue
    #     left_max_x = max([x1, x2])
    #     right_min_x = min([x1, x2])
    #     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    return img_add

if __name__== '__main__':
    
    start = time()
    
    input_path = '/home/ma-user/work/sourcecode/input_video/inputvideo.mp4'
    output_path = '/home/ma-user/work/sourcecode/output/laneDetection.mp4'
    videoCapture = cv2.VideoCapture(input_path)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_num = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoWrite = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    success, frame = videoCapture.read()
    count = 1
    for count in tqdm(range(1, int(frame_num + 1))):
        success, frame = videoCapture.read()
        if success:
            image = get_car_lane(frame)
            videoWrite.write(image)  # 写视屏
    videoWrite.release()
    end = time()
    print(end - start)
    # img = cv2.imread("d:/desktop/lane1.jpg")
    # img_add = get_car_lane(img)
    # cv2.imshow('im', img_add)
    # cv2.waitKey(0)
    