from typing import List, Tuple

import cv2
import numpy as np
import math
from numpy.linalg import solve, inv
a_x = 0
a_y = 0
def main():
    originalImage = cv2.imread('exemplo3.jpg')

    grayImage = getGrayImage(originalImage)
    # imageShowWithWait("grayImage", grayImage)

    edgeImage = getEdgeImage(grayImage)
    # imageShowWithWait("edgeImage", edgeImage)

    lineImage, lineY = getAxisLinesImage(originalImage, edgeImage)
    # imageShowWithWait("lineImage", lineImage)
    #
    withoutAxisImage = getImageWithoutXYAxis(edgeImage)
    # imageShowWithWait("withoutAxisImage", withoutAxisImage)
    #
    paraboleImage = getParabolaImage(lineImage, withoutAxisImage)
    imageShowWithWait("paraboleImage", paraboleImage)
    cv2.waitKey(100000)

def getGrayImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = 7
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1)

def getEdgeImage(image):
    return cv2.threshold(image, 143, 255, cv2.THRESH_BINARY_INV)[1]

def getAxisLinesImage(originalImage, edgeImage):
    lineX, lineY = getAxisXY(edgeImage, originalImage)
    cv2.line(originalImage, lineX[0], lineX[1], (255, 0, 0), 2)
    cv2.line(originalImage, lineY[0], lineY[1], (255, 0, 0), 2)
    return originalImage, lineY


def getAxisXY(edgeImage, originalImage):
    lines = getAxisLinesFromImage(edgeImage)
    lineX, lineY = getAxisWithMLS(lines, originalImage)
    return lineX, lineY


def getAxisLinesFromImage(edgeImage):
    rho = 4  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 1 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 300  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edgeImage, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    return lines

def getAxisWithMLS(lines, originalImage):
    axisX = []
    axisY = []
    maxSizeY = 0
    maxSizeX = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            if ((x1 - x2)**2 > (y1 - y2)**2):
                axisX.append(line)
            else:
                axisY.append(line)
    biggestLineSize = 0
    biggestLine = None
    for yLine in axisY:
        x1, y1, x2, y2 = yLine[0]
        lineSize = sizeOfLine((x1, y1), (x2, y2))
        if( lineSize > biggestLineSize):
            biggestLineSize = lineSize
            biggestLine = yLine
    _, biggestLineY1, _, biggestLineY2 = biggestLine[0]
    validAxisY = []
    for yLine in axisY:
        x1, y1, x2, y2 = yLine[0]
        if not (math.fabs(y1 - biggestLineY1) > 100 and math.fabs(y2 - biggestLineY2) > 100 ):
            validAxisY.append(yLine)
    # for line in validAxisY:
    #     cv2.line(originalImage, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 0, 0), 2)
    # for line in axisX:
    #     color = (255, 0, 0)
    #     cv2.line(originalImage, (line[0][0], line[0][1]), (line[0][2], line[0][3]), color, 2)
    # imageShowWithWait("asd",originalImage)
    lineX =  getPredictedLine(axisX, "x")
    lineY = getPredictedLine(validAxisY, "y")
    return lineX, lineY

def getPredictedLine(lines, axis):
    A = []
    B = []
    global a_x, a_y
    for line in lines:
        for x1, y1, x2, y2 in line:
            A.append([x1, 1])
            A.append([x2, 1])
            B.append([y1])
            B.append([y2])

    a, b = LSM(A,B)
    if axis == "x":
        a_x = a
    else:
        a_y = a

    return [(0, int(b[0])), (1500, int(a[0]*1500 + b[0]))]

def getImageWithoutXYAxis(edgeImage):
    line_image = np.copy(edgeImage) * 0  # creating a blank to draw lines on
    lines = getAxisXY(edgeImage, np.copy(edgeImage))
    line: List[Tuple[int, int]]
    for line in lines:
        (x1,y1), (x2, y2) = line
        cv2.line(edgeImage, (x1, y1), (x2, y2), (0, 0, 0), 80)
    return cv2.addWeighted(edgeImage, 0.8, line_image, 1, 0)

def getParabolaImage(originalImage, edgeImage):
    rho = 4  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

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

    C = solveParableLSM(points)

    a, b, c = C
    points = []
    print(-a," * x^2 + (", -b, " * x) + (", -c, ")" )

    drawParable(a, b, c, originalImage, points)
    return originalImage


def drawParable(a, b, c, originalImage, points):
    NoneType = type(None)
    for x in range(0, 1500, 1):
        x1, x2 = bhaskara(a, b, c - x)
        if not isinstance(x1, NoneType):
            points = points + [(x1, x)]
            points = points + [(x2, x)]
        y = a*x*x + b*x + c
        if y > 0:
            points = points + [(x,int((y)))]
    ang = np.arctan(a_x)
    for x, y in points:
        x, y = rotate(x, y, ang)
        point = (int(x), int(y))
        cv2.circle(originalImage, point, 1, (0, 255, 252))


def solveParableLSM(points):
    A = []
    B = []
    ang = np.arctan(a_x)
    for point in points:
        x = point[0]
        y = point[1]
        x, y = rotate(x, y, -ang)
        A.append([x * x, x, 1])
        B.append(y)
    C = LSM(A, B)
    return C


def LSM(A, B):
    A = np.array(A)
    B = np.array(B)
    tranposeA = A.transpose()
    X = tranposeA.dot(A)
    C = solve(X, tranposeA.dot(B))
    return C


def bhaskara(a, b, c):
    delta = (b ** 2) - (4 * a * c)
    if (delta < 0):
        return (None, None)
    x = math.sqrt(delta)
    x1 = (-b + x) / (2 * a)
    x2 = (-b - x) / (2 * a)
    return (int(x1), int(x2))

def rotate(x, y, rad):
    xx = math.cos(rad) * x - math.sin(rad) * y
    yy = math.sin(rad) * x + math.cos(rad) * y
    return (xx, yy)

def sizeOfLine(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    deltaX = (x1 - x2) * (x1 - x2)
    deltaY = (y1 - y2) * (y1 - y2)
    return math.sqrt(deltaX + deltaY)

def imageShowWithWait(windowName, image):
    cv2.imshow(windowName, image)
    cv2.waitKey(100)

if __name__ == '__main__':
    main()
