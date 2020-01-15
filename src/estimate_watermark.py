import math
import cv2
import numpy
import numpy as np
import scipy
import scipy.fftpack
from src.preprocess import plot_images

# Variables
KERNEL_SIZE = 3


def estimate_watermark(images):
    """
    Given a folder, estimate the watermark (grad(W) = median(grad(J)))
    Also, give the list of gradients, so that further processing can be done on it
    """

    # Compute gradients
    print("Computing images gradients.")
    grad_x = np.array([cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE) for x in images.values()])
    grad_y = np.array([cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE) for x in images.values()])

    # Compute median of grads
    print("Computing median gradients.")
    Wm_x = np.median(grad_x, axis=0)
    Wm_y = np.median(grad_y, axis=0)

    num_images = len(grad_x)

    return Wm_x, Wm_y, num_images


def crop_watermark_ignore_borders(grad_x, grad_y, threshold=0.4, boundary_size=2):

    grad_x, grad_y = crop_watermark(grad_x, grad_y, threshold=0.4, boundary_size=-5)
    grad_x, grad_y = crop_watermark(grad_x, grad_y, threshold=threshold, boundary_size=boundary_size)

    return grad_x, grad_y


def crop_watermark(grad_x, grad_y, threshold=0.4, boundary_size=2):
    """
    Crops the watermark by taking the edge map of magnitude of grad(W)
    Assumes the grad_x and grad_y to be in 3 channels
    @param: threshold - gives the threshold param
    @param: boundary_size - boundary around cropped images
    """
    W_mod = np.sqrt(np.square(grad_x) + np.square(grad_y))
    W_mod = to_plot_normalize_image(W_mod)
    W_gray = threshold_image(np.average(W_mod, axis=2), threshold=threshold)
    x, y = np.where(W_gray == 1)

    xm, xM = np.min(x) - boundary_size - 1, np.max(x) + boundary_size + 1
    ym, yM = np.min(y) - boundary_size - 1, np.max(y) + boundary_size + 1
    xm, ym, xM, yM = max(xm, 0), max(ym, 0), min(xM, W_gray.shape[0]), min(yM, W_gray.shape[1])
    # plot_images([W_gray, W_gray[xm:xM, ym:yM]], False)

    return grad_x[xm:xM, ym:yM, :], grad_y[xm:xM, ym:yM, :]


def poisson_reconstruct(grad_x, grad_y, kernel_size=KERNEL_SIZE, num_iters=100, h=0.1,
                        boundary_image=None, boundary_zero=True):
    """
    Iterative algorithm for Poisson reconstruction.
    Given the grad_x and grad_y values, find laplacian, and solve for images
    Also return the squared difference of every step.
    h = convergence rate
    """
    fxx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=kernel_size)
    fyy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # fyy = grad_y[1:, :-1] - grad_y[:-1, :-1]
    # fxx = grad_x[:-1, 1:] - grad_x[:-1, :-1]
    laplacian = fxx + fyy
    m, n, p = laplacian.shape

    if boundary_zero:
        est = np.zeros(laplacian.shape)
    else:
        assert (boundary_image is not None)
        assert (boundary_image.shape == laplacian.shape)
        est = boundary_image.copy()

    est[1:-1, 1:-1, :] = np.random.random((m - 2, n - 2, p))
    loss = []

    for i in range(num_iters):
        old_est = est.copy()
        est[1:-1, 1:-1, :] = 0.25 * (
                est[0:-2, 1:-1, :] + est[1:-1, 0:-2, :] + est[2:, 1:-1, :] + est[1:-1, 2:, :] - h * h * laplacian[
                                                                                                        1:-1, 1:-1,
                                                                                                        :])
        error = np.sum(np.square(est - old_est))
        loss.append(error)

    return est


def poisson_reconstruct2(grad_x, grad_y, boundarysrc=None):
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # Laplacian
    gyy = grad_y[1:, :-1] - grad_y[:-1, :-1]
    gxx = grad_x[:-1, 1:] - grad_x[:-1, :-1]

    if boundarysrc is None:
        boundarysrc = numpy.zeros(grad_y.shape)

    f = numpy.zeros(boundarysrc.shape)
    f[:-1, 1:] += gxx
    f[1:, :-1] += gyy

    # Boundary images
    boundary = boundarysrc.copy()
    boundary[1:-1, 1:-1] = 0

    # Subtract boundary contribution
    f_bp = -4 * boundary[1:-1, 1:-1] + boundary[1:-1, 2:] + boundary[1:-1, 0:-2] + boundary[2:, 1:-1] + \
        boundary[0:-2, 1:-1]
    f = f[1:-1, 1:-1] - f_bp

    # Discrete Sine Transform
    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

    # Eigenvalues
    (x, y) = numpy.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = (2 * numpy.cos(math.pi * x / (f.shape[1] + 2)) - 2) + (2 * numpy.cos(math.pi * y / (f.shape[0] + 2)) - 2)

    f = numpy.divide(fsin, denom[:, :, None])

    # Inverse Discrete Sine Transform
    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

    # New center + old boundary
    result = boundary
    result[1:-1, 1:-1] = img_tt

    return result


def get_cropped_images(images, cropped_Wm_x, cropped_Wm_y, wm_thr=0.05, img_thr=0.1, trash_thr=0.02):
    """
    This is the part where we get all the images, extract their parts, and then add it to our matrix
    """
    # images_cropped = np.zeros((num_images,) + shape)
    images_cropped = {}

    for file, img in images.items():
        img_marked, wm_start, wm_end = watermark_detector(img, cropped_Wm_x, cropped_Wm_y,
                                                          wm_thr=wm_thr,
                                                          img_thr=img_thr,
                                                          trash_thr=trash_thr,
                                                          )
        img_result = img[max(wm_start[0], 0):wm_start[0] + wm_end[0], max(wm_start[1], 0):wm_start[1] + wm_end[1], :]

        if not img_result.any():
            continue

        images_cropped[file] = img_result

        # images_cropped[index, :, :, :] = _img

    return images_cropped


def watermark_detector(img, gx, gy, wm_thr=0.05, img_thr=0.1, trash_thr=0.02, printval=False):
    """
    Compute a verbose edge map using Canny edge detector, take its magnitude.
    Assuming cropped values of gradients are given.
    Returns images, start and end coordinates
    """
    Wm = (np.average(np.sqrt(np.square(gx) + np.square(gy)), axis=2))
    Wm = threshold_image(Wm, threshold=wm_thr)
    # plot_images([Wm], False)
    
    img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE)
    img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE)
    img_g = (np.average(np.sqrt(np.square(img_x) + np.square(img_y)), axis=2))
    img_edgemap = threshold_image(img_g, threshold=img_thr, trash_thr=trash_thr, invert=True)
    # plot_images([img_g], False)
    
    # img_edgemap = (cv2.Canny(img[:,:,0], thresh_low, thresh_high))
    chamfer_dist = cv2.filter2D(img_edgemap, -1, Wm)

    if printval:
        plot_images([img_edgemap], False)

    rect = Wm.shape
    index = np.unravel_index(np.argmax(chamfer_dist), img.shape[:-1])

    x, y = int(index[0] - rect[0] / 2), int(index[1] - rect[1] / 2)  # Must be an integer, so make adjustments
    im = img.copy()
    cv2.rectangle(im, (y, x), (y + rect[1], x + rect[0]), (255, 0, 0))

    return im, (x, y), (rect[0], rect[1])


def to_plot_normalize_image(image):
    """
    to_plot_normalize_image: Give a normalized images matrix which can be used with implot, etc.
    Maps to [0, 1]
    """
    img = image.astype(float)
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def threshold_image(image, threshold=0.5, trash_thr=0.01, invert=False):
    """
    Threshold the images to make all its elements greater than threshold*MAX = 1
    """
    img = to_plot_normalize_image(image)

    if invert:
        img[img >= threshold] = 0
        img[img > trash_thr] = 1
    else:
        img[img >= threshold] = 1
        img[img < 1] = 0

    return img


def normalized(image):
    """
    Return the images between -1 to 1 so that its easier to find out things like
    correlation between images, convolutionss, etc.
    Currently required for Chamfer distance for template matching.
    """
    return 2 * to_plot_normalize_image(image) - 1
