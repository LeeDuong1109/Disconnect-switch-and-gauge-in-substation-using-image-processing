import cv2
import numpy as np
from skimage.feature import canny
from skimage import morphology
import function as ff

horizontal_coeffect = 122
# img_path = '30.png'
# path = 'DCL' + '/' + img_path
def horizontal_test(img_path):
    img = cv2.imread('DCL' + '/' + img_path, 1)
    print("mean: {0}\nstd: {1}\n".format(np.median(img), np.std(img)))
    # img = cv2.resize(img, (412, int((412/img.shape[1])*img.shape[0])), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_img = np.zeros(img.shape[:2])
    black_img2 = np.zeros(img.shape[:2])

    kernel = np.ones((3, 3), np.uint8)
    # filter = np.array(
    #         [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
    img_blur = cv2.medianBlur(img, 5)
    thresh_img = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY)[1]

    # print("mean: ", np.mean(thresh_img))
    # print("std: ", np.std(thresh_img))
    # Copy the thresholded image.

    thresh_img = cv2.dilate(thresh_img, kernel, iterations=2)
    thresh_img = cv2.erode(thresh_img, kernel, iterations=3)
    thresh_img = cv2.bilateralFilter(thresh_img, 9, 75, 75)
    thresh_img = cv2.dilate(thresh_img, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if contours is not None:
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
            if len(approx) < 30:
                cv2.drawContours(thresh_img, [approx], -1, (255, 255, 255), 6)
            # if len(approx) < 10:
            #     cv2.drawContours(horizontal, [approx], -1, (0, 0, 0), 3)
                # cv2.drawContours(img_or, [approx], -1, (0, 255, 0), 2)
    # thresh_img = cv2.erode(thresh_img, kernel, iterations=1)
    thresh_img = cv2.dilate(thresh_img, kernel, iterations=1)

    im_floodfill = thresh_img.copy()
    h, w = thresh_img.shape[:2]
    mask1 = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask1, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    contours, _ = cv2.findContours(im_floodfill_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if contours is not None:
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
            if len(approx) < 30:
                cv2.drawContours(im_floodfill_inv, [approx], -1, (255, 255, 255), 6)
            # if len(approx) < 10:
            #     cv2.drawContours(horizontal, [approx], -1, (0, 0, 0), 3)
            # cv2.drawContours(img_or, [approx], -1, (0, 255, 0), 2)

    # thresh_img = thresh_img | im_floodfill_inv

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(255-thresh_img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 100 #600
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    mask = np.zeros(img2.shape[:2])
    mask = img2.astype('uint8')
    dest_and = cv2.bitwise_and(img, mask, mask=mask)

    canny_auto = ff.auto_canny(img, sigma=0.33)
    canny_auto = cv2.dilate(canny_auto, kernel, iterations=1)
    canny_auto = cv2.bitwise_and(canny_auto, mask, mask=None)
    canny_auto = cv2.dilate(canny_auto, kernel, iterations=1)

    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask2 = cv2.morphologyEx(canny_auto, cv2.MORPH_CLOSE, se1)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, se2)

    # mask = np.dstack([mask, mask, mask]) / 255
    path = 'DCL_mask' + '/' + img_path
    cv2.imwrite(path, mask2)
    # # canny_auto = canny_auto*mask
    #
    # horizontal = np.copy(canny_auto)
    # cols = horizontal.shape[1]
    # horizontal_size = cols // horizontal_coeffect #30
    # horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # horizontal = cv2.erode(horizontal, horizontalStructure)
    # horizontal = cv2.dilate(horizontal, horizontalStructure)
    # # horizontal = cv2.dilate(horizontal, kernel, iterations=3)
    # # horizontal = cv2.erode(horizontal, kernel, iterations=1)
    # # horizontal = cv2.dilate(horizontal,kernel, iterations=1)
    # contours, _ = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # if contours is not None:
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
    #         x_c = approx.ravel()[0]
    #         y_c = approx.ravel()[1]
    #         if len(approx) > 50:
    #             cv2.drawContours(black_img, [approx], -1, (255, 255, 255), 8)
    #         # if len(approx) < 10:
    #         #     cv2.drawContours(horizontal, [approx], -1, (0, 0, 0), 3)
    #             # cv2.drawContours(img_or, [approx], -1, (0, 255, 0), 2)
    #
    # minLineLength = 0
    # maxLineGap = 40
    # lines = cv2.HoughLinesP(canny_auto, 1, np.pi / 180, 70, minLineLength, maxLineGap)
    # if lines is not None:
    #     for x in range(0, len(lines)):
    #         for x1, y1, x2, y2 in lines[x]:
    #             if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 40:
    #                 cv2.line(canny_auto, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # canny_auto = canny_auto.astype('uint8')
    # # canny_auto = cv2.erode(canny_auto, kernel, iterations=1)
    #
    # minLineLength = 0
    # maxLineGap = 40
    # lines = cv2.HoughLinesP(canny_auto, 1, np.pi/180, 75, minLineLength, maxLineGap) #70
    # if lines is not None:
    #     for x in range(0, len(lines)):
    #         for x1, y1, x2, y2 in lines[x]:
    #             if np.sqrt((x1-x2)**2+(y1-y2)**2) > 40:
    #                 cv2.line(black_img2, (x1, y1),(x2, y2),(255, 255, 255), 3)
    # black_img2 = black_img2.astype('uint8')
    # black_img2 = cv2.erode(black_img2, kernel, iterations=1)
    #
    # rho = 1  # distance resolution in pixels of the Hough grid
    # theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    # min_line_length = 0  # minimum number of pixels making up a line
    # max_line_gap = 30  # maximum gap in pixels between connectable line segments
    # # line_image = np.copy(img) * 0  # creating a blank to draw lines on
    #
    # # Run Hough on edge detected image
    # # Output "lines" is an array containing endpoints of detected line segments
    # lines = cv2.HoughLinesP(black_img2, rho, theta, threshold, np.array([]),
    #                         min_line_length, max_line_gap)
    # if lines is not None:
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 40:
    #                 cv2.line(black_img2, (x1, y1), (x2, y2), (255, 255, 255), 2)
    #
    # black_img2 = cv2.erode(black_img2, kernel, iterations=1)
    # black_img2 = cv2.Sobel(black_img2, cv2.CV_64F, 0, 1, ksize=1)  # y
    # # black_img2 = cv2.erode(black_img2, kernel, iterations=1)
    # black_img2 = cv2.dilate(black_img2, kernel, iterations=1)
    # print(canny_auto.dtype, black_img2.dtype)
    # black_img2 = black_img2.astype('uint8')
    #
    # contours, _ = cv2.findContours(black_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # if contours is not None:
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
    #         x_c = approx.ravel()[0]
    #         y_c = approx.ravel()[1]
    #         if len(approx) < 10:
    #             cv2.drawContours(black_img2, [approx], -1, (0, 0, 0), 3)
    # # black_img = cv2.erode(black_img, kernel, iterations=0)
    # black_img2 = cv2.Sobel(black_img2, cv2.CV_64F, 0, 1, ksize=1)  # y
    # black_img2 = black_img2.astype('uint8')
    #
    # lines = cv2.HoughLinesP(black_img2, rho, theta, threshold, np.array([]),
    #                         min_line_length, max_line_gap)
    # if lines is not None:
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 15:
    #                 cv2.line(black_img2, (x1, y1+1), (x2, y2+1), (255, 255, 255), 2)
    # # black_img2 = cv2.blur(black_img2, (5, 5))
    # # black_img2 = cv2.GaussianBlur(black_img2, (5, 5), 0)
    # black_img2 = cv2.medianBlur(black_img2, 5)
    #
    # horizontal = np.copy(black_img2)
    # cols = horizontal.shape[1]
    # horizontal_size = 1 #cols // 140  # 30
    # horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # horizontal = cv2.erode(horizontal, horizontalStructure)
    # horizontal = cv2.dilate(horizontal, horizontalStructure)
    # # horizontal = cv2.dilate(horizontal, kernel, iterations=3)
    # # horizontal = cv2.erode(horizontal, kernel, iterations=1)
    # # horizontal = cv2.dilate(horizontal,kernel, iterations=1)
    # contours, _ = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # if contours is not None:
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
    #         x_c = approx.ravel()[0]
    #         y_c = approx.ravel()[1]
    #         if len(approx) > 6:
    #             cv2.drawContours(black_img2, [approx], -1, (255, 255, 255), 2)
    # black_img2 = cv2.erode(black_img2, kernel, iterations=2)
    #
    # path = 'DCL_horizital' + '/' + img_path
    # cv2.imwrite(path, black_img2)
    # path = 'DCL_canny' + '/' + img_path
    # cv2.imwrite(path, canny_auto)
    path = 'DCL_results' + '/' + img_path
    cv2.imwrite(path, dest_and)
    path = 'flood_fill' + '/' + img_path
    cv2.imwrite(path, im_floodfill_inv)