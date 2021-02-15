import cv2
import numpy as np
from numpy import random
import os
import imutils

def add_boder(image_path, output_path, low, high):
    """
    low: kích thước biên thấp nhất (pixel)
    hight: kích thước biên lớn nhất (pixel)
    """

    # random các kích thước biên trong khoảng (low, high)
    top = random.randint(low, high)
    bottom = random.randint(low, high)
    left = random.randint(low, high)
    right = random.randint(low, high)
    image = cv2.imread(image_path)
    original_width, original_height = image.shape[1], image.shape[0]
    # sử dụng hàm của opencv để thêm biên
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    # sau đó resize ảnh bằng kích thước ban đầu của ảnh
    image = cv2.resize(image, (original_width, original_height))
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(output_path, image)

def random_crop(image_path, out_path):
    image = cv2.imread(image_path)
    original_width, original_height = image.shape[1], image.shape[0]
    x_center, y_center = original_height // 2, original_width // 2

    x_left = random.randint(0, x_center // 2)
    x_right = random.randint(original_width - x_center // 2, original_width)

    y_top = random.randint(0, y_center // 2)
    y_bottom = random.randint(original_height - y_center // 2, original_width)

    # crop ra vùng ảnh với kích thước ngẫu nhiên
    cropped_image = image[y_top:y_bottom, x_left:x_right]
    # resize ảnh bằng kích thước ảnh ban đầu
    cropped_image = cv2.resize(cropped_image, (original_width, original_height))
    # cv2.imshow('image', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(out_path, cropped_image)

def change_brightness(image_path, value):
    """
    value: độ sáng thay đổi
    """
    hsv = cv2.cvtColor(image_path, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


def decrease_brightness(image_path, value):
    """
    value: độ sáng thay đổi
    """
    # img = cv2.imread(image_path)
    hsv = cv2.cvtColor(image_path, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.subtract(v, value)
    v[v > 255] = 255
    v[v < 0] = 0

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(output_path, img)
    return img

def rotate_image(image_path, range_angle, output_path):
    """

    :param image_path:
    :param range_angle:
    :param output_path:
    :return:
    """
    """
    range_angle: Khoảng góc quay
    """
    image = cv2.imread(image_path)
    #lựa chọn ngẫu nhiên góc quay
    angle = random.randint(-range_angle, range_angle)
    img_rot = imutils.rotate(image, angle)
    cv2.imwrite(output_path, img_rot)
    # cv2.imshow('image', img_rot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u

def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    # print(v)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def auto_gammar(image, sigma=0.33):
    # v = np.median(image)
    # print(v)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def median2dp(median):
    # print(int((72/175)*median), median)
    return int((72/175)*(median/10))