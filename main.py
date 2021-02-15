import cv2
import numpy as np
# from matplotlib import pyplot as plt
# from sphinx.ext.ifconfig import ifconfig
import function as ff

path = 'DCL/26.png'

img = cv2.imread(path, 1)
img1 = cv2.imread(path, 1)
img2 = img1.copy()

img = np.array(255 * (img / 255) ** 2.2, dtype='uint8')
# img = ff.decrease_brightness(img, 50)
# img = ff.change_brightness(img1, 105)
img = cv2.bilateralFilter(img, 7, 75, 75)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/cdf.max()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

img = cdf[img]

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/cdf.max()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

img = cdf[img]
# img = np.array(img)

# img = (img-img.min())/(img.max()-img.min())

# cv2.imshow('image', img)
# cv2.imshow('image1', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = ff.decrease_brightness(img, 5)
# img = cv2.bilateralFilter(img, 7, 15, 15)
# cv2.imshow("output", img1)
# cv2.waitKey(0)
# img = ff.decrease_brightness(img, 115)

# img = ff.change_brightness(img1, 15)
# img = cv2.Laplacian(img, cv2.CV_32F)

# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# kernel = np.ones((5, 5),np.float32)/25
# img = cv2.filter2D(img, -1, kernel)

# img = ff.change_brightness(img1, 1)
# img = ff.decrease_brightness(img1, 5)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # detect circles in the image
#
# edges = cv2.Canny(img, 130, 550)
# cv2.imshow("output", edges)
# cv2.waitKey(0)
#
# circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.2, 100)
# # ensure at least some circles were found
# if circles is not None:
# 	# convert the (x, y) coordinates and radius of the circles to integers
# 	circles = np.round(circles[0, :]).astype("int")
# 	# loop over the (x, y) coordinates and radius of the circles
# 	for (x, y, r) in circles:
# 		# draw the circle in the output image, then draw a rectangle
# 		# corresponding to the center of the circle
# 		cv2.circle(img1, (x, y), r, (0, 255, 0), 4)
# 		cv2.rectangle(img1, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
# 	# show the output image
# 	cv2.imshow("output", img1)
# 	cv2.waitKey(0)
# else:
# 	print("nothing")

# edges = cv2.Canny(img, 120, 550)
# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 100)
contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(contours)
a = []

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
    a.append(len(approx))

for contour in contours:
    # print(len(contour))
    # cv2.imshow('image1',contours)
    approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
    # print(len(approx))
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) >= 70:
        # print(approx)
        cv2.drawContours(img1, [approx], 0, (0, 0, 0), 2)
    if len(approx) == max(a) and len(approx) >= 95:

        cv2.circle(img1, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img1, "Circle", (x, y), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                   minRadius=5, maxRadius=90)

edges = cv2.Canny(img, 100, 550)

lines = cv2.HoughLinesP(image=edges, rho=3, theta=np.pi / 180, threshold=15, minLineLength=15, maxLineGap=1)

final_line_list = []
diff1LowerBound = 0.15 #diff1LowerBound and diff1UpperBound determine how close the line should be from the center
diff1UpperBound = 0.25
diff2LowerBound = 0.5 #diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
diff2UpperBound = 1.0

# lines = cv2.HoughLinesP(image=img, rho=3, theta=np.pi / 180, threshold=100,minLineLength=10, maxLineGap=0)


if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(img2, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(img2, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # show the output image

    j = 0
    if lines is not None:
        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                # print(x1, y1, x2, y2)
                if np.sqrt((x - x1) ** 2 + (y - y1) ** 2) < r / 2:
                    cv2.line(img2, (x, y), (x1, y1), (0, 255, 255), 2)
                else:
                    j += 1
                    if j == len(lines):
                        print("nothing")
    else:
        print("nothing")

else:
	print("nothing")


cv2.imshow('image', img)
cv2.imshow('image1', img1)
cv2.imshow('edges', edges)
cv2.imshow("output", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()