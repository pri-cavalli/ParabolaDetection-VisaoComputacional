import cv2
import numpy as np

def main():
    originalImage = cv2.imread('exemplo1.jpg')

    grayImage = getGrayImage(originalImage)
    # imageShowWithWait("grayImage", grayImage)

    edgeImage = getEdgeImage(grayImage)
    # imageShowWithWait("edgeImage", edgeImage)

    lineImage = getLinesXYImage(originalImage, edgeImage)
    # imageShowWithWait("lineImage", lineImage)

    withoutAxisImage = getImageWithoutXYAxis(edgeImage)
    # imageShowWithWait("withoutAxisImage", withoutAxisImage)

    paraboleImage = getParabolaImage(lineImage, withoutAxisImage)
    imageShowWithWait("paraboleImage", paraboleImage)


def getGrayImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = 7
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

def getEdgeImage(image):
    return cv2.Canny(image, 100, 100)

def getLinesXYImage(originalImage, edgeImage):
    rho = 4  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 300  # minimum number of pixels making up a line
    max_line_gap = 129  # maximum gap in pixels between connectable line segments
    line_image = np.copy(originalImage) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edgeImage, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return cv2.addWeighted(originalImage, 0.8, line_image, 1, 0)

def getImageWithoutXYAxis(edgeImage):
    rho = 4  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 300  # minimum number of pixels making up a line
    max_line_gap = 129  # maximum gap in pixels between connectable line segments
    line_image = np.copy(edgeImage) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edgeImage, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(edgeImage, (x1, y1), (x2, y2), (0, 0, 0), 15)
    return cv2.addWeighted(edgeImage, 0.8, line_image, 1, 0)

def getParabolaImage(originalImage, edgeImage):
    rho = 4  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 49  # maximum gap in pixels between connectable line segments
    line_image = np.copy(originalImage) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edgeImage, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return cv2.addWeighted(originalImage, 0.8, line_image, 1, 0)

def imageShowWithWait(image, windowName):
    cv2.imshow(image, windowName)
    cv2.waitKey(1000)

if __name__ == '__main__':
    main()
