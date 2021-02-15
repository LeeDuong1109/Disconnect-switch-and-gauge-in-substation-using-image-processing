import cv2
import numpy as np

import cv2
import numpy as np
from skimage.feature import canny

import function as ff

# Function to map each intensity level to output intensity level.
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1) * pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

    # Open the image.

dis = []
def nothing(x):
    dis = []
    pass

path = 'DCL/3.png'
img = cv2.imread(path, 0)
img3 = cv2.imread(path, 1)
img_or = cv2.imread(path, 1)
black_img = np.zeros(img.shape)

# Open the image.
cv2.namedWindow("Tracking")
cv2.namedWindow("Tracking1")
cv2.createTrackbar("LH", "Tracking", 54, 400, nothing)
cv2.createTrackbar("r1", "Tracking", 68, 255, nothing)
cv2.createTrackbar("s1", "Tracking", 0, 255, nothing)
cv2.createTrackbar("r2", "Tracking", 66, 255, nothing)
cv2.createTrackbar("s2", "Tracking", 0, 255, nothing)
cv2.createTrackbar("m", "Tracking1", 10, 255, nothing)
cv2.createTrackbar("thres", "Tracking", 0, 255, nothing)
cv2.createTrackbar("db", "Tracking", 20, 255, nothing)
cv2.createTrackbar("dp", "Tracking", 44, 255, nothing)
cv2.createTrackbar("min radius", "Tracking", 39, 255, nothing)
cv2.createTrackbar("max radius", "Tracking", 72, 255, nothing)
cv2.createTrackbar("minDist", "Tracking", 255, 400, nothing)
cv2.createTrackbar("canny1", "Tracking", 20, 255, nothing)
cv2.createTrackbar("canny2", "Tracking", 20, 255, nothing)
cv2.createTrackbar("sigma_s", "Tracking", 10, 255, nothing)
cv2.createTrackbar("sigma_r", "Tracking", 20, 255, nothing)
cv2.createTrackbar("threshold", "Tracking", 20, 255, nothing)
cv2.createTrackbar("minLineLength", "Tracking", 20, 255, nothing)
cv2.createTrackbar("maxLineGap", "Tracking", 1, 255, nothing)
cv2.createTrackbar("bilateralFilter", "Tracking1", 4, 255, nothing)
cv2.createTrackbar("iterations_dilate", "Tracking1", 1, 255, nothing)
cv2.createTrackbar("iterations_erode", "Tracking1", 3, 255, nothing)
cv2.createTrackbar("iterations_dilate_1", "Tracking1", 0, 255, nothing)
cv2.createTrackbar("iterations_erode_1", "Tracking1", 0, 255, nothing)
cv2.createTrackbar("iterations_dilate_2", "Tracking1", 0, 255, nothing)
cv2.createTrackbar("iterations_erode_2", "Tracking1", 0, 255, nothing)
cv2.createTrackbar("iterations_dilate_3", "Tracking1", 4, 255, nothing)
cv2.createTrackbar("iterations_erode_3", "Tracking1", 10, 255, nothing)
cv2.createTrackbar("g", "Tracking1", 0, 255, nothing)
cv2.createTrackbar("contour", "Tracking1", 77, 200, nothing)
cv2.createTrackbar("contour_1", "Tracking1", 51, 200, nothing)
#threshold=15, minLineLength=15, maxLineGap=1
# Trying 4 gamma values.
# for gamma in [0.1, 0.5, 1.2, 2.2]:
# Apply gamma correction.

kernel = np.ones((5, 5), np.uint8)
while(1):
    img_or = cv2.imread(path, 1)
    img = cv2.imread(path, 1)
    black_img = np.zeros(cv2.imread(path, 0).shape)
    # img_large = cv2.imread(path, 1)
    median = (np.median(img))
    # print(median)

    bilateral = cv2.getTrackbarPos("bilateralFilter", "Tracking1")*2 + 1
    img = cv2.bilateralFilter(img, bilateral, 175, 175)
    m = cv2.getTrackbarPos("m", "Tracking1")
    img = cv2.medianBlur(img, 2*m+1)
    sigma_s = cv2.getTrackbarPos("sigma_s", "Tracking")
    sigma_r = cv2.getTrackbarPos("sigma_r", "Tracking")/100

    # img = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    img = cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)
    img = cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_large = img.copy()
    # img = cv2.erode(img, kernel, iterations=1)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.erode(img, element, iterations=1)
    mask = cv2.dilate(mask, element, iterations=1)

    iterations_dilate = cv2.getTrackbarPos("iterations_dilate", "Tracking1")
    iterations_erode = cv2.getTrackbarPos("iterations_erode", "Tracking1")
    iterations_dilate_1 = cv2.getTrackbarPos("iterations_dilate_1", "Tracking1")
    iterations_erode_1 = cv2.getTrackbarPos("iterations_erode_1", "Tracking1")
    iterations_dilate_2 = cv2.getTrackbarPos("iterations_dilate_2", "Tracking1")
    iterations_erode_2 = cv2.getTrackbarPos("iterations_erode_2", "Tracking1")
    iterations_dilate_3 = cv2.getTrackbarPos("iterations_dilate_3", "Tracking1")
    iterations_erode_3 = cv2.getTrackbarPos("iterations_erode_3", "Tracking1")
    g = cv2.getTrackbarPos("g", "Tracking1")


    img = cv2.dilate(img, kernel, iterations=iterations_dilate)
    img = cv2.erode(img, kernel, iterations=iterations_erode)

    img = cv2.dilate(img, kernel, iterations=iterations_dilate_1)
    img = cv2.erode(img, kernel, iterations=iterations_erode_1)


    # img_large = cv2.dilate(img_large.copy(), cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=iterations_dilate_3)
    # img_large = cv2.erode(img_large.copy(), cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=iterations_erode_3)

    # img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

    # kernel = np.array([[1, 2, 1],
    #                    [0, 0, 0],
    #                    [-1, -2, -1]])
    #
    # # kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
    #
    # # filter the source image
    # img = cv2.filter2D(img, -1, kernel)

    # img = cv2.dilate(img, kernel, iterations=iterations_dilate_2)
    # img = cv2.erode(img, kernel, iterations=iterations_erode_2)

    # kernel = np.array([[0.0, -1.0, 0.0],
    #                    [-1.0, 4.0, -1.0],
    #                    [0.0, -1.0, 0.0]])
    #
    # kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
    # # kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
    #
    # # filter the source image
    # img = cv2.filter2D(img, -1, kernel)

    img = 255 - cv2.GaussianBlur(img, (2*g+1, 2*g+1), 0)
    # img_large = 255 - cv2.GaussianBlur(img_large, (2*g+1, 2*g+1), 0)

    # blur = cv2.bilateralFilter(img, 9, 75, 75)
    # img = ff.decrease_brightness(img, 20)
    #######################################################
    thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)

    # edges_morph = ff.auto_canny(morph, sigma=0.33)
    ###########################################################################
    # image_yuv = cv2.cvtColor(cv2.imread('26.png', 1), cv2.COLOR_BGR2YUV)
    img3 = cv2.imread(path, 0)
    # dest_and_2 = cv2.bitwise_and(img3, img, mask=None)
    # image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    # image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    # image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_RGB2GRAY)
    ################################################################
    # Creating maxican hat filter
    filter = np.array(
        [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])

    # filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # filter = np.array([[3, -2, -3], [-4, 8, -6], [5, -1, -0]])
    # filter = np.array(
    #     [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])

    # Applying cv2.filter2D function on our Logo image
    # mexican_hat_img2 = cv2.filter2D(img, -1, filter)
    # edges_mexican = ff.auto_canny(mexican_hat_img2, sigma=0.33)
    #################################################################
    l_h = cv2.getTrackbarPos("LH", "Tracking")/10
    r1 = cv2.getTrackbarPos("r1", "Tracking")
    s1 = cv2.getTrackbarPos("s1", "Tracking")
    r2 = cv2.getTrackbarPos("r2", "Tracking")
    s2 = cv2.getTrackbarPos("s2", "Tracking")
    thres = cv2.getTrackbarPos("thres", "Tracking")
    db = cv2.getTrackbarPos("db", "Tracking")
    dp = cv2.getTrackbarPos("dp", "Tracking")/10
    min = cv2.getTrackbarPos("min radius", "Tracking")
    max = cv2.getTrackbarPos("max radius", "Tracking")
    minDist = cv2.getTrackbarPos("minDist", "Tracking")
    canny1 = cv2.getTrackbarPos("canny1", "Tracking")
    canny2 = cv2.getTrackbarPos("canny2", "Tracking")
    threshold = cv2.getTrackbarPos("threshold", "Tracking")
    minLineLength = cv2.getTrackbarPos("minLineLength", "Tracking")
    maxLineGap = cv2.getTrackbarPos("maxLineGap", "Tracking")
    cntt = cv2.getTrackbarPos("contour", "Tracking1")
    cntt_1 = cv2.getTrackbarPos("contour_1", "Tracking1")
    # threshold=15, minLineLength=15, maxLineGap=1
    #####################################################################
    # img = cv2.medianBlur(img, 2*m + 1)
    # ret, img = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
    ###############################################################################
    gamma_corrected = np.array(255 * (img / 255) ** l_h, dtype='uint8')
    # gamma_corrected_large = np.array(255 * (img_large / 255) ** l_h, dtype='uint8')

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    img1 = cdf[img]

    hist, bins = np.histogram(img1.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    img1 = cdf[img]

    canny = cv2.Canny(gamma_corrected, canny1, canny2)
    canny_auto = ff.auto_canny(img1, sigma=0.33)

    mexican_hat_img2 = cv2.filter2D(img1, -1, filter)
    # r1 = 70
    # s1 = 0
    # r2 = 140
    # s2 = 255

    # Vectorize the function to apply it to each value in the Numpy array.
    pixelVal_vec = np.vectorize(pixelVal)

    # Apply contrast stretching.
    img2 = pixelVal_vec(gamma_corrected, r1, s1, r2, s2)

    # img2_large = pixelVal_vec(gamma_corrected_large, r1, s1, r2, s2)
    # dest_and_2 = cv2.bitwise_and(img2, img, mask=None)
    # print(img2.shape, img3.shape)
    # print(img2.shape)
    # img22 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # lines = None
    # ret, thresh = cv2.threshold(img, 0, 255, 0)

    dest_and = cv2.bitwise_and(gamma_corrected, img3, mask=None)
    # canny_auto = cv2.bitwise_and(hist, canny_auto, mask=None)
    # print(np.var(dest_and))
    # print(img.shape, canny_auto.shape)


    # canny_auto = cv2.Sobel(canny_auto, cv2.CV_64F, 0, 1, ksize=5)
    # canny_auto = np.absolute(canny_auto)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    bw = cv2.adaptiveThreshold(mexican_hat_img2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 15, -2)
    bw = cv2.bitwise_and(bw, img, mask=None)
    # bw = cv2.dilate(bw, kernel, iterations=1)

    horizontal = np.copy(bw)
    cols = horizontal.shape[1]
    horizontal_size = cols // cntt #30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # horizontal = cv2.dilate(horizontal, kernel, iterations=1)
    # horizontal = cv2.erode(horizontal, kernel, iterations=1)

    contours, _ = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours = contours[0] if len(contours) <= 5 else contours[1]
    # contours = contours[0]

    # for c in contours:
    #     area = cv2.contourArea(c)
    #     if area < 1000:
    #         x, y, w, h = cv2.boundingRect(c)
    #         img_or[y:y + h, x:x + w] = img_or[y:y + h, x:x + w]

    if contours is not None:
        for contour in contours:
            # print(len(contour))
            # cv2.imshow('image1',contours)
            area = cv2.contourArea(contour)
            # print(area)
            approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
            # print(len(approx))
            x_c = approx.ravel()[0]
            y_c = approx.ravel()[1]
            # dis.append(np.sqrt((cx-x_c)**2 + (cy-y_c)**2))
            # if np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2) < 7 * r / 8:
            # if len(approx) > 40:
            #     cv2.drawContours(mexican_hat_img2, [approx], 0, (255, 255, 255), 4)
            # else:
            #     cv2.drawContours(mexican_hat_img2, [approx], 0, (0, 0, 0), 4)
            if len(approx) > cntt_1:
                cv2.drawContours(black_img, [approx], -1, (255, 255, 255), 2)
                cv2.drawContours(img_or, [approx], -1, (0, 255, 0), 2)
            # cv2.polylines(dest_and, cnt, True, (0, 0, 255), 2)
            # cv2.circle(canny_auto, (x, y), 5, (0, 0, 255), -1)
    else:
        print("duong")

    black_img = cv2.dilate(black_img, kernel, iterations=iterations_dilate_2)
    black_img = cv2.erode(black_img, kernel, iterations=iterations_erode_2)

    M = cv2.moments(black_img)
    try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(img_or, (cx, cy), 5, (0, 0, 255), 5)
    except:
        pass

    # canny_auto = cv2.bitwise_and(canny_auto, mexican_hat_img2, mask=None)


    """
    blur = img_or.copy()
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    remove_vertical = cv2.morphologyEx(thres, cv2.MORPH_OPEN, vertical_kernel)
    """

    # dis = []
    # circles = cv2.HoughCircles(img1, method=cv2.HOUGH_GRADIENT, dp=ff.median2dp(median), minDist=minDist,
    #                            minRadius=min, maxRadius=max)

    # if circles is not None:
    #     # convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
    #     # loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         # draw the circle in the output image, then draw a rectangle
    #         # corresponding to the center of the circle
    #         cv2.circle(img3, (x, y), r, (0, 255, 0), 4)
    #         cv2.circle(img3, (x, y), 3, (0, 255, 255), 7)
    #         # cv2.rectangle(img3, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #     # show the output image
    #     else:
    #         # print("nothing")
    #         pass
    # else:
    #     # print("nothing")
    #     pass


    # lines = cv2.HoughLinesP(image=mexican_hat_img2, rho=3, theta=np.pi / 180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         for x1, y1, x2, y2 in lines[i]:
    #             # print(x1, y1, x2, y2)
    #             # if np.sqrt((x - x1) ** 2 + (y - y1) ** 2) < r:
    #             cv2.line(img_or, (x2, y2), (x1, y1), (255, 0, 255), 2)
    # Save edited images.

    # cv2.imshow('power_law', gamma_corrected)
    cv2.imshow('original', img)
    # cv2.imshow('hist', img1)
    # cv2.imshow('piecewise_linear', img2)
    # cv2.imshow('image_circle', img3)
    # cv2.imshow('image_rgb', image_rgb)
    # cv2.imshow('canny', canny)
    cv2.imshow('canny_auto', canny_auto)
    # cv2.imshow('edges_morph', thresh_img)
    cv2.imshow('mexican_hat', mexican_hat_img2)
    cv2.imshow('dest_and', dest_and)
    cv2.imshow('bw', bw)
    cv2.imshow('horizontal', horizontal)
    cv2.imshow('original_BRG', img_or)
    cv2.imshow('black', black_img)
    # cv2.imshow('thresh', thres)
    # cv2.imshow('dest_and_2', dest_and_2)

    # cv2.imshow('mask', mask)
    # cv2.imshow('detailEnhance', dst)
    # cv2.imshow('detailEnhance1', img_dst)
    # cv2.imshow('Bilateral Filtering', blur)
    # cv2.imshow('mexican_hat_edges', edges_mexican)
    key = cv2.waitKey(1)
    if key == 27:
        # print(np.var(dis))
        break

cv2.destroyAllWindows()