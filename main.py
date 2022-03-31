from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import argparse


def optimize_contours(cnts, delta):
    better_cnt = []
    for c in cnts:
        was = []
        for new_point in c:
            bad = False
            for prev_point in was:
                if ((new_point[0][0] - prev_point[0][0])**2 + (new_point[0][1] - prev_point[0][1])**2)**0.5 < delta:
                    bad = True
                    break
            if not bad:
                was.append(new_point)
        better_cnt.append(np.array(was))
    return better_cnt


def get_best(cnts):
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return approx

    cnt = cnts[0]
    return cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)


def draw_edges(image_name, path='', thresholds=(40, 100), blur=11,
               scale=10, take=10, delta=5, optimize=True,
               show_edges=False, show_result=True):
    sizes = (10, 15)

    image = cv2.imread(path + image_name)
    original = image.copy()
    image = cv2.resize(image, (image.shape[1]//10, image.shape[0]//10))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (blur, blur), 0)
    edged = cv2.Canny(blured, thresholds[0], thresholds[1])

    if show_edges:
        plt.figure(figsize=sizes)
        plt.imshow(edged, cmap='gray')
        plt.show()

    cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    if optimize:
        cnts = optimize_contours(cnts, delta)
    
    cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:take]
    paper_contour = get_best(cnts)

    cv2.drawContours(original, [paper_contour * scale], -1, (255, 0, 0), 15)
    if show_result:
        plt.figure(figsize=sizes)
        plt.imshow(original)
        plt.show()

    return paper_contour * scale, original


def set_argparse():
    parser = argparse.ArgumentParser(description='Find paper edges on image.')
    parser.add_argument('name', metavar='image_name', type=str,
                        help='name of the image ot process')
    parser.add_argument('--threshold1', type=int, default=40,
                        help='threshold1 for Canny algorithm (default: 40)')
    parser.add_argument('--threshold2', type=int, default=100,
                        help='threshold2 for Canny algorithm (default: 100)')
    parser.add_argument('--blur', type=int, default=11,
                        help='blur for Canny algorithm (default: 11)')
    parser.add_argument('--scale', type=int, default=10,
                        help='scale for image compress (scale: 10)')
    parser.add_argument('--take', type=int, default=10,
                        help='how many images from sorted list to take (default: 10)')
    parser.add_argument('--dist', type=int, default=5,
                        help='distance for deleting extra points (default: 5)')
    parser.add_argument('--optimize', type=bool, default=True,
                        help='need to use optimize for search points (default: True)')
    parser.add_argument('--dont_show_edges', action='store_const', const=True, default=False,
                        help='show edges (default: False)')
    parser.add_argument('--show_result', action='store_const', const=False, default=True,
                        help='dont show result image (default: True)')
    parser.add_argument('--save_name', type=str, default=None,
                        help='name to save image (default: None)')
    return parser

def main(args):
    contour, img = draw_edges(args.name, thresholds=(args.threshold1, args.threshold2),
                              blur=args.blur, scale=args.scale, take=args.take,
                              delta=args.dist, optimize=args.optimize,
                              show_edges=args.dont_show_edges, show_result=args.show_result)

    if args.save_name is not None:
        plt.imshow(img)
        plt.savefig(args.save_name)


if __name__ == '__main__':
    parser = set_argparse()
    main(parser.parse_args())
