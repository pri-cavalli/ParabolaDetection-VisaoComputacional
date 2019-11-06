import cv2
import numpy as np
import math
from numpy.linalg import solve, inv

def main():
    originalImage = cv2.imread('exemplo1.jpg')

    grayImage = getGrayImage(originalImage)
    # imageShowWithWait("grayImage", grayImage)

    edgeImage = getEdgeImage(grayImage)
    # imageShowWithWait("edgeImage", edgeImage)

    lineImage, lineY = getAxisLinesImage(originalImage, edgeImage)
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

def getAxisLinesImage(originalImage, edgeImage):
    lines = getAxisLinesFromImage(originalImage, edgeImage)
    lineX, lineY = getAxisWithMLS(lines)
    cv2.line(originalImage, lineX[0], lineX[1], (255, 0, 0), 2)
    cv2.line(originalImage, lineY[0], lineY[1], (255, 0, 0), 2)
    return originalImage, lineY

def getAxisLinesFromImage(originalImage, edgeImage):
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
    return lines

def getAxisWithMLS(lines):
    axisX = []
    axisY = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if ((x1 - x2)**2 > (y1 - y2)**2):
                axisX.append(line)
            else:
                axisY.append(line)
    return getPredictedLine(axisX, "x"), getPredictedLine(axisY, "y")

def getPredictedLine(lines, axis):
    X = []
    Y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            X = X + [float(x1), float(x2)]
            Y = Y + [y1, y2]
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    num = 0
    den = 0
    for i in range(len(X)):
        num += (X[i] - X_mean) * (Y[i] - Y_mean)
        den += (X[i] - X_mean) ** 2
    m = num / den
    c = Y_mean - m * X_mean
    Y_pred = m * np.asarray(X) + c
    if (axis == "x"):
        return [(int(min(X)), int(np.mean(Y_pred))), (int(max(X)), int(np.mean(Y_pred)))]
    return [(int(np.mean(X)), int(min(Y_pred))), (int(np.mean(X)), int(max(Y_pred)))]

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

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edgeImage, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.circle(originalImage, (x1, y1), 5, (0, 255, 0))
            cv2.circle(originalImage, (x2, y2), 5, (0, 255, 0))
            points += [(x1, y1), (x2, y2)]
    A = []
    B = []
    for point in points:
        x = point[0]
        y = point[1]
        A.append([x * x, x, 1])
        B.append(y)
    A = np.array(A)
    B = np.array(B)
    tranposeA = A.transpose()
    X = tranposeA.dot(A)
    C = solve(X, tranposeA.dot(B))

    a, b, c = C
    points = []

    NoneType = type(None)
    for i in range(0, 756, 5):
        x1, x2 = bhaskara(a, b, c - i)
        if not isinstance(x1, NoneType):
            points = points + [(x1, i)]
            points = points + [(x2, i)]
    print(points)
    for point in points:
        cv2.circle(originalImage, point, 5, (0, 255, 252))
    return originalImage


def bhaskara(a, b, c):
    delta = (b ** 2) - (4 * a * c)
    if (delta < 0):
        return (None, None)
    x = math.sqrt(delta)
    x1 = (-b + x) / (2 * a)
    x2 = (-b - x) / (2 * a)
    return (int(x1), int(x2))
    return originalImage

def imageShowWithWait(image, windowName):
    cv2.imshow(image, windowName)
    cv2.waitKey(8000)

if __name__ == '__main__':
    main()
