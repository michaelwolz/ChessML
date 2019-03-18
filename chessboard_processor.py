import math
import operator
import sys
from collections import defaultdict

import numpy as np
import cv2

import scipy.spatial as spatial
import scipy.cluster as clstr


def canny(img):
    # Maybe I'll add some auto thresholding here
    edges = cv2.Canny(img, 80, 200)
    return edges


def hough_lines(img):
    rho, theta, thresh = 2, np.pi / 180, 600
    return cv2.HoughLines(img, rho, theta, thresh)


def sort_lines(lines):
    """
    Sorts lines by horizontal and vertical
    """
    h = []
    v = []
    for i in range(lines.shape[0]):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v.append([rho, theta])
        else:
            h.append([rho, theta])
    return h, v


def calculate_intersections(h, v):
    """
    Finds the intersection of two lines given in Hesse normal form.
    See https://stackoverflow.com/a/383527/5087436
    """
    points = []
    for rho1, theta1 in h:
        for rho2, theta2 in v:
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            point = np.linalg.solve(A, b)
            point = int(np.round(point[0])), int(np.round(point[1]))
            points.append(point)
    return np.array(points)


def cluster_intersections(points, max_dist=40):
    # I want to change this to kmeans
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), clusters)

    result = []
    for point in clusters:
        result.append([point[0], point[1]])
    return result


def find_chessboard_corners(points):
    """
    Code from https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2
    """
    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value
    bottom_right, _ = max(enumerate([pt[0] + pt[1] for pt in points]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0] + pt[1] for pt in points]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0] - pt[1] for pt in points]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0] - pt[1] for pt in points]), key=operator.itemgetter(1))
    return [points[top_left], points[top_right], points[bottom_left], points[bottom_right]]


def distance_between(p1, p2):
    """
    Code from https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2
    """
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def warp_image(img, edges):
    """
    Code from https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2
    """
    top_left, top_right, bottom_left, bottom_right = edges[0], edges[1], edges[2], edges[3]

    # Explicitly set the data type to float32 or 'getPerspectiveTransform' will throw an error
    warp_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    # Describe a square with side of the calculated length, this is the new perspective we want to warp to
    warp_dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(warp_src, warp_dst)

    # Performs the transformation on the original image
    return cv2.warpPerspective(img, m, (int(side), int(side)))


def cut_chessboard(img, output_path, output_prefix=""):
    side_len = int(img.shape[0] / 8)
    for i in range(8):
        for j in range(8):
            tile = img[i * side_len: (i + 1) * side_len, j * side_len: (j + 1) * side_len]
            cv2.imwrite(output_path + output_prefix + "-" + str(j + i * 8) + ".jpg", tile)


def resize_image(img):
    """
    Resizes image to a maximum width of 800px
    """
    width = img.shape[1]
    if width > 800:
        scale = 800 / width
        return cv2.resize(img, None, fx=scale, fy=scale)
    else:
        return img


def process_chessboard(src_path, output_path, output_prefix="", debug=False):
    src = cv2.imread(src_path)

    if src is None:
        sys.exit("There is no file with this path!")

    src = resize_image(src)
    src_copy = src.copy()

    # Convert to grayscale
    process = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    if debug:
        cv2.imshow("Grayscale", process)
        cv2.waitKey()
        cv2.destroyWindow("Grayscale")

    # Blur to remove disturbing things
    process = cv2.blur(process, (4, 4))

    if debug:
        cv2.imshow("Blur", process)
        cv2.waitKey()
        cv2.destroyWindow("Blur")

    # Use Canny Edge Detector https://en.wikipedia.org/wiki/Canny_edge_detector
    process = canny(process)

    if debug:
        cv2.imshow("Canny", process)
        cv2.waitKey()
        cv2.destroyWindow("Canny")

    # Dilate image (thicker lines)
    process = cv2.dilate(process, np.ones((3, 3), dtype=np.uint8))

    if debug:
        cv2.imshow("Dilate", process)
        cv2.waitKey()
        cv2.destroyWindow("Dilate")

    # Use Hough transform to detect lines https://en.wikipedia.org/wiki/Hough_transform
    lines = hough_lines(process)

    # Sort lines by horizontal and vertical
    h, v = sort_lines(lines)

    if debug:
        render_lines(src_copy, h, (0, 255, 0))
        render_lines(src_copy, v, (0, 0, 255))
        cv2.imshow("Sorted lines", src_copy)
        cv2.waitKey()
        cv2.destroyWindow("Sorted lines")

    if len(h) < 9 or len(v) < 9:
        print("There are not enough horizontal and vertical lines in this image. Try it anyway!")

    # Calculate intersections of the horizontal and vertical lines
    intersections = calculate_intersections(h, v)

    if debug:
        render_intersections(src_copy, intersections, (255, 0, 0), 1)
        cv2.imshow("Intersections", src_copy)
        cv2.waitKey()
        cv2.destroyWindow("Intersections")

    # Cluster intersection since there are many
    clustered = cluster_intersections(intersections)

    if debug:
        src_copy = src.copy()
        render_intersections(src_copy, clustered, (255, 0, 0), 5)
        cv2.imshow("Clustered Intersections", src_copy)
        cv2.waitKey()
        cv2.destroyWindow("Clustered Intersections")

    if len(clustered) != 81:
        print("Something is wrong. There are " + str(len(intersections)) + " instead of 81 intersections.")

    # Find outer corners of the chessboard
    corners = find_chessboard_corners(clustered)

    if debug:
        src_copy = src.copy()
        render_intersections(src_copy, corners, (255, 0, 0), 5)
        cv2.imshow("Corners", src_copy)
        cv2.waitKey()
        cv2.destroyWindow("Corners")

    # Warp and crop image
    dst = warp_image(src, corners)

    if debug:
        cv2.imshow("Warped", dst)
        cv2.waitKey()
        cv2.destroyWindow("Warped")

    # Cut chessboard into its 64 tiles
    cut_chessboard(dst, output_path, output_prefix)


def render_lines(img, lines, color):
    for rho, theta in lines:
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)


def render_intersections(img, points, color, size):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color, size)


def main():
    # TODO: instead of checking if there are enough intersections and lines, check if the corners are right.
    process_chessboard('data/chessboards/processed/IMG_3269.jpg', "data/out/", "", True)


if __name__ == "__main__":
    main()
