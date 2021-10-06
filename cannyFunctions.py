
import numpy as np


def convolve2d(image, kernel):
    output = np.zeros(image.shape, image.dtype)
    ker_rows = kernel.shape[0]
    # set padding size depending on kernel size
    pad_size = kernel.shape[0] // 2
    if pad_size == 0:
        pad_size = 1
    # Add zero padding to the image
    padded = np.zeros((image.shape[0] + 2 * pad_size, image.shape[1] + 2 * pad_size))
    padded[pad_size:-pad_size, pad_size:-pad_size] = image
    # for every pixel multiply the kernel and the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum((kernel * padded[i: i+ker_rows, j: j+ker_rows]))
    return output


def gradient_intensity(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
    dx = convolve2d(img, kernel_x)
    dy = convolve2d(img, kernel_y)
    d = np.arctan2(dy, dx)
    return dx, d


def round_angle(angle):
    angle = np.rad2deg(angle) % 180
    if 0 <= angle < 22.5 or 157.5 <= angle < 180:
        angle = 0
    elif 22.5 <= angle < 67.5:
        angle = 45
    elif 67.5 <= angle < 112.5:
        angle = 90
    elif 112.5 <= angle < 157.5:
        angle = 135
    return angle


def suppression(img, d):
    m, n = img.shape
    z = np.zeros((m, n), dtype=np.int32)
    for i in range(m):
        for j in range(n):
        # find neighbour pixels to visit from the gradient directions
            where = round_angle(d[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        z[i, j] = img[i, j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        z[i, j] = img[i, j]
                elif where == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        z[i, j] = img[i, j]
                elif where == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        z[i, j] = img[i, j]
            except IndexError as e:
                """ Todo: Deal with pixels at the image boundaries. """
                pass
    return z


def threshold(img, t, tt):
    # define gray value of a WEAK and a STRONG pixel
    cf = {'WEAK': np.int32(50), 'STRONG': np.int32(255)}
    # get strong pixel indices
    strong_i, strong_j = np.where(img > tt)
    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= tt))
    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)
    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)
    return img, cf.get('WEAK')


def tracking(img, weak, strong=255):
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                            or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                            or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def canny(image, kernel, low_thresh=50, high_thresh=90):
    smoothed7 = convolve2d(image, kernel)
    smoothed7x2 = convolve2d(smoothed7, kernel)
    gradient, d = gradient_intensity(smoothed7x2)
    suppressed = suppression(np.copy(gradient), d)
    th, weak = threshold(np.copy(suppressed), low_thresh, high_thresh)
    tracked = tracking(np.copy(th), weak)

    return tracked

