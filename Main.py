import cv2
import numpy as np
import math

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

    paraboleImage = getParabolaImage(lineImage, withoutAxisImage, lineY)
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

def getParabolaImage(originalImage, edgeImage, lineY):
    rho = 4  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 49  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edgeImage, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    print(lineY)
    for x1, y1, x2, y2 in lines[0]:
        point1 = [x1, y1]
        cv2.circle(originalImage, (x1, y1), 5, (0, 0, 255), -1)
    for x1, y1, x2, y2 in lines[3]:
        point2 = [x1, y1]
        cv2.circle(originalImage, (x1, y1), 5, (0, 0, 255), -1)
    for x1, y1, x2, y2 in lines[48]:
        point3 = [x2, y2]
        cv2.circle(originalImage, (x2, y2), 5, (0, 0, 255), -1)
    a, b, c = calc_parabola_vertex(point1[0], point1[1], point2[0], point2[1], point3[0], point3[1])
    print(point1, point2, point3)
    points = []
    if lineY[0][1] > lineY[1][1]:
        max = lineY[0][1]
        min = lineY[1][1]
    else:
        min = lineY[0][1]
        max = lineY[1][1]
    for i in range(min, max, 5):
        x1, x2 = bhaskara(a, b, c-i)
        points = points + [(x1, i)]
        points = points + [(x2, i)]
    print(a, b, c)
    print(min)
    print(max)
    print(points)
    for point in points:
        cv2.circle(originalImage, point, 5, (0, 255, 0))
    return originalImage

def bhaskara( a, b, c):
    delta = (b ** 2) - (4 * a * c)
    if ( delta < 0 ):
        return (None,None)
    x = math.sqrt( delta )
    x1 = (-b + x) / (2 * a)
    x2 = (-b - x) / (2 * a)
    return (int(x1), int(x2))

def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
    return A, B, C

def imageShowWithWait(image, windowName):
    cv2.imshow(image, windowName)
    cv2.waitKey(8000)

if __name__ == '__main__':
    main()
