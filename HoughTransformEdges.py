import cv2
import numpy as np
import cannyFunctions
from matplotlib import pyplot as plt

WIDTH = 600
HEIGHT = 600
rec_width = 300
rec_height = 420


def create_rectangle(img, x1, x2, y1, y2, color):
    image = cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return image


def add_noise(img):
    mean, std = cv2.meanStdDev(img)
    noise = np.random.normal(mean, std, size=img.shape)  # size = how many draws
    # apply noise and normalize values
    noisy_image = np.clip((img + noise*0.1), 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def gradient_intensity(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
    dx = cannyFunctions.convolve2d(img, kernel_x)
    dy = cannyFunctions.convolve2d(img, kernel_y)
    g = np.hypot(dx, dy)
    d = np.arctan2(dy, dx)
    return (g, d)


def canny(image, kernel, low_thresh=50, high_thresh=90):
    smoothed7 = cannyFunctions.convolve2d(image, kernel)
    smoothed7x2 = cannyFunctions.convolve2d(smoothed7, kernel)
    gradient, d = gradient_intensity(np.asarray(smoothed7x2,dtype="int32"))
    suppressed = cannyFunctions.suppression(gradient, d)
    th, weak = cannyFunctions.threshold(suppressed, low_thresh, high_thresh)
    tracked = cannyFunctions.tracking(th, weak)
    return tracked


def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height))) # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos


def get_rect_points(acc, thetas, rhos):
    points = list()
    minimum = np.argmin(acc) - 1
    # get 8 maximum values to draw 4 lines (two points each)
    for i in range(8):
        idx = np.argmax(acc)
        rho = rhos[idx // acc.shape[1]]
        theta = thetas[idx % acc.shape[1]]
        acc[idx // acc.shape[1], idx % acc.shape[1]] = minimum
        points.append((rho, theta))
    return points


def convert_points(points, img):
    for rho, theta in points:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # calculate x and y points according to rectangle
        x1 = int(x0 + 450 * (-b))
        y1 = int(y0 + 570 * a)
        x2 = int(x0 + 150 * (-b))
        y2 = int(y0 + 150 * a)
        cv2.line(img, (x2, y2), (x1, y1), (255, 255, 255), 2)
    return img


def main():
    img = np.zeros((WIDTH, HEIGHT))
    x = int(rec_width / 2)
    y = int(HEIGHT / 4)
    xx = x + rec_width
    yy = y + rec_height
    col = (255, 255, 255)
    img = create_rectangle(img, x, xx, y, yy, col)

    # add noise to image
    noisy_image = add_noise(img)
    gaussian7x7 = np.array([[1, 6, 15, 20, 15, 6, 1],
                            [6, 36, 90, 120, 90, 36, 6],
                            [15, 90, 225, 300, 225, 90, 15],
                            [20, 120, 300, 400, 300, 120, 20],
                            [15, 90, 225, 300, 225, 90, 15],
                            [6, 36, 90, 120, 90, 36, 6],
                            [1, 6, 15, 20, 15, 6, 1]]) / 4096
    # perform canny algorithm on noisy image
    cannied = canny(noisy_image, gaussian7x7, 100, 255)
    # perform hough transform
    acc, thetas, rhos = hough_line(cannied)
    # get polar points and convert them to cartesian
    points = get_rect_points(acc, thetas, rhos)
    lines = convert_points(points, np.copy(cannied))
    # plot images
    imgs = [(img, 'image'), (noisy_image, 'noisy'), (cannied, 'noisy canny'), (lines, 'hough')]
    for i in range(len(imgs)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(imgs[i][0], 'gray', vmin=0, vmax=255)
        plt.title(imgs[i][1])
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
