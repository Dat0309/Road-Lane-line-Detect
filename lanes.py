import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag

def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    #get height of image
    height = image.shape[0]
    poligon = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    #fill mask in poligon with color white
    cv.fillPoly(mask, poligon, 255)
    # Thực hiện phép "and" hai mảng nhị phân của hai hình ảnh để lọc ra viền của poligon
    mask_image = cv.bitwise_and(image, mask)
    return mask_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def average_slope_intercept(image, lines):
    left_fir= []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fir.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    lef_fit_average = np.average(left_fir, axis=0)
    right_fit_average = np.average(right_fit, axis= 0)
    left_line = make_coordinates(image, lef_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
     
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/ slope)
    x2 = int((y2 - intercept)/ slope)
    return np.array([x1, y1, x2, y2])

# image = cv.imread('test_image.jpg')
# lane_image = np.copy(image)
# canny_img =  canny(lane_image)
# lines = cv.HoughLinesP(region_of_interest(canny_img), 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)

# combo_image = cv.addWeighted(lane_image, 0.8, line_image, 1, 1)

# cv.imshow("image", combo_image)
# cv.waitKey(0)
# plt.imshow(canny)
# plt.show()

cap = cv.VideoCapture('test2.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    canny_img =  canny(frame)
    lines = cv.HoughLinesP(region_of_interest(canny_img), 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)

    combo_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)

    cv.imshow("video", combo_image)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()