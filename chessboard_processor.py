import math
import operator
from collections import defaultdict

import numpy as np
import cv2

import scipy.spatial as spatial
import scipy.cluster as clstr



"""
Thanks to https://medium.com/@daylenyang/building-chess-id-99afa57326cd
and https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2
"""


def canny(img):
    # Maybe I'll add some auto thresholding here
    edges = cv2.Canny(img, 100, 200)
    return edges


def hough_lines(img):
    rho, theta, thresh = 2, np.pi / 180, 650
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


def cluster_intersections(points, max_dist=50):
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


def cut_chessboard_into_pieces_this_is_its_last_resort(img):
    side_len = int(img.shape[0] / 8)
    for i in range(8):
        for j in range(8):
            print("Extracting tile:", j + i * 8)
            tile = img[i * side_len: (i + 1) * side_len, j * side_len: (j + 1) * side_len]
            cv2.imwrite("data/out/" + str(j + i * 8) + ".jpg", tile)


def process_chessboard(src_path):
    src = cv2.imread(src_path)

    # Convert to grayscale
    process = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Blur to remove disturbing things
    process = cv2.blur(process, (4, 4))

    # Use Canny Edge Detector https://en.wikipedia.org/wiki/Canny_edge_detector
    process = canny(process)

    # Dilate image (thicker lines)
    process = cv2.dilate(process, np.ones((3, 3), dtype=np.uint8))

    # Use Hough transform to detect lines
    lines = hough_lines(process)

    # Sort lines by horizontal and vertical
    h, v = sort_lines(lines)

    # Calculate intersections of the horizontal and vertical lines
    intersections = calculate_intersections(h, v)

    # Cluster intersection since there are many
    cluster = cluster_intersections(intersections)

    # Find outer corners of the chessboard
    corners = find_chessboard_corners(cluster)

    # Warp image
    dst = warp_image(src, corners)

    # Cut chessboard into its 64 tiles
    cut_chessboard_into_pieces_this_is_its_last_resort(dst)

    # Just some visualisations for development
    # render_lines(h, (0, 0, 255))
    # render_lines(v, (0, 255, 0))
    # render_intersections(edges, (255, 0, 0))
    # cv2.imshow("Source", src)
    # cv2.waitKey()


def render_lines(img, lines, color):
    for rho, theta in lines:
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)


def render_intersections(img, points, color):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color, 5)


def main():
    process_chessboard('data/chess-sera.jpeg')


if __name__ == "__main__":
    main()
