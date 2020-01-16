from scipy.sparse import linalg
from src.closed_form_matting import *
from src.estimate_watermark import *
from matplotlib import pyplot as plt


def estimate_normalized_alpha(J, W_m, threshold=170, invert=False, adaptive=True, adaptive_threshold=21,
                              c2=10):
    num_images = len(J)
    m, n, _ = J[0].shape

    # _Wm = (255 * to_plot_normalize_image(np.average(W_m, axis=2))).astype(np.uint8)
    _Wm = (np.average(W_m, axis=2)).astype(np.uint8)
    plot_images([_Wm], False)

    if adaptive:
        thr = cv2.adaptiveThreshold(_Wm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_threshold, c2)
    else:
        _, thr = cv2.threshold(_Wm, threshold, 255, cv2.THRESH_BINARY)

    if invert:
        thr = 255 - thr

    # plot_images([thr], False)

    thr = np.stack([thr, thr, thr], axis=2)

    print(f'Estimating normalized alpha using {num_images} images')
    alpha = np.array([closed_form_matte(img, thr) for img in J])

    return np.median(alpha, axis=0), thr


def estimate_blend_factor(J, W_m, alpha, threshold=0.01 * 255):
    K = len(J)

    J = np.array(J)
    Jm = J - W_m
    gx_jm = np.array([cv2.Sobel(Jm[i], cv2.CV_64F, 1, 0, 3) for i in range(K)])
    gy_jm = np.array([cv2.Sobel(Jm[i], cv2.CV_64F, 0, 1, 3) for i in range(K)])

    Jm_grad = np.sqrt(gx_jm ** 2 + gy_jm ** 2)

    est_Ik = alpha * np.median(J, axis=0)
    gx_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 1, 0, 3)
    gy_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 0, 1, 3)
    estIk_grad = np.sqrt(gx_estIk ** 2 + gy_estIk ** 2)

    C = [
        np.sum(Jm_grad[:, :, :, i] * estIk_grad[:, :, i]) / np.sum(np.square(estIk_grad[:, :, i])) / K
        for i in [0, 1, 2]
    ]

    return C, est_Ik


def solve_images(J, W_m, alpha, W_init, gamma=1, beta=1, lambda_w=0.005, lambda_i=1, lambda_a=0.01, iters=4):
    """
    Master solver, follows the algorithm given in the supplementary.
    W_init: Initial value of W
    Step 1: Image Watermark decomposition
    """
    # prepare variables
    K, m, n, p = J.shape
    size = m * n * p

    sobelx = get_xSobel_matrix(m, n, p)
    sobely = get_ySobel_matrix(m, n, p)
    Ik = np.zeros(J.shape)
    Wk = np.zeros(J.shape)
    for i in range(K):
        Ik[i] = J[i] - W_m
        Wk[i] = W_init.copy()

    # This is for median images
    W = W_init.copy()

    # Iterations
    for _ in range(iters):

        print("------------------------------------")
        print("Iteration: %d" % (_))

        # Step 1
        print("Step 1")
        alpha_gx = cv2.Sobel(alpha, cv2.CV_64F, 1, 0, 3)
        alpha_gy = cv2.Sobel(alpha, cv2.CV_64F, 0, 1, 3)

        Wm_gx = cv2.Sobel(W_m, cv2.CV_64F, 1, 0, 3)
        Wm_gy = cv2.Sobel(W_m, cv2.CV_64F, 0, 1, 3)

        cx = diags(np.abs(alpha_gx).reshape(-1))
        cy = diags(np.abs(alpha_gy).reshape(-1))

        alpha_diag = diags(alpha.reshape(-1))
        alpha_bar_diag = diags((1 - alpha).reshape(-1))

        for i in range(K):
            # prep vars
            Wkx = cv2.Sobel(Wk[i], cv2.CV_64F, 1, 0, 3)
            Wky = cv2.Sobel(Wk[i], cv2.CV_64F, 0, 1, 3)

            Ikx = cv2.Sobel(Ik[i], cv2.CV_64F, 1, 0, 3)
            Iky = cv2.Sobel(Ik[i], cv2.CV_64F, 0, 1, 3)

            alphaWk = alpha * Wk[i]
            alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)
            alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)

            phi_data = diags(Func_Phi_deriv(np.square(alpha * Wk[i] + (1 - alpha) * Ik[i] - J[i]).reshape(-1)))
            # phi_W = diags(Func_Phi_deriv(np.square(np.abs(alpha_gx) * Wkx + np.abs(alpha_gy) * Wky).reshape(-1)))
            # phi_I = diags(Func_Phi_deriv(np.square(np.abs(alpha_gx) * Ikx + np.abs(alpha_gy) * Iky).reshape(-1)))
            phi_f = diags(Func_Phi_deriv(((Wm_gx - alphaWk_gx) ** 2 + (Wm_gy - alphaWk_gy) ** 2).reshape(-1)))
            phi_aux = diags(Func_Phi_deriv(np.square(Wk[i] - W).reshape(-1)))
            phi_rI = diags(Func_Phi_deriv(np.abs(alpha_gx) * (Ikx ** 2) + np.abs(alpha_gy) * (Iky ** 2)).reshape(-1))
            phi_rW = diags(Func_Phi_deriv(np.abs(alpha_gx) * (Wkx ** 2) + np.abs(alpha_gy) * (Wky ** 2)).reshape(-1))

            L_i = sobelx.T.dot(cx * phi_rI).dot(sobelx) + sobely.T.dot(cy * phi_rI).dot(sobely)
            L_w = sobelx.T.dot(cx * phi_rW).dot(sobelx) + sobely.T.dot(cy * phi_rW).dot(sobely)
            L_f = sobelx.T.dot(phi_f).dot(sobelx) + sobely.T.dot(phi_f).dot(sobely)
            A_f = alpha_diag.T.dot(L_f).dot(alpha_diag) + gamma * phi_aux

            bW = alpha_diag.dot(phi_data).dot(J[i].reshape(-1)) + beta * L_f.dot(W_m.reshape(-1)) + gamma * phi_aux.dot(
                W.reshape(-1))
            bI = alpha_bar_diag.dot(phi_data).dot(J[i].reshape(-1))

            A = vstack([hstack(
                [(alpha_diag ** 2) * phi_data + lambda_w * L_w + beta * A_f, alpha_diag * alpha_bar_diag * phi_data]), \
                hstack([alpha_diag * alpha_bar_diag * phi_data,
                        (alpha_bar_diag ** 2) * phi_data + lambda_i * L_i])]).tocsr()

            b = np.hstack([bW, bI])

            phi_f = None
            alphaWk = None
            phi_aux = None
            L_i = None
            phi_rI = None
            L_w = None
            L_f = None
            phi_rW = None
            A_f = None
            phi_data = None

            x = linalg.spsolve(A, b)

            Wk[i] = x[:size].reshape(m, n, p)
            Ik[i] = x[size:].reshape(m, n, p)
            plt.subplot(3, 1, 1)
            plt.imshow(to_plot_normalize_image(J[i]))
            plt.subplot(3, 1, 2)
            plt.imshow(to_plot_normalize_image(Wk[i]))
            plt.subplot(3, 1, 3)
            plt.imshow(to_plot_normalize_image(Ik[i]))
            plt.draw()
            plt.pause(0.001)
            print(i)

        # Step 2
        print("Step 2")
        W = np.median(Wk, axis=0)

        # plt.imshow(to_plot_normalize_image(W))
        # plt.draw()
        # plt.pause(0.001)

        # Step 3
        print("Step 3")
        W_diag = diags(W.reshape(-1))

        for i in range(K):
            alphaWk = alpha * Wk[i]
            alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)
            alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)
            phi_f = diags(Func_Phi_deriv(((Wm_gx - alphaWk_gx) ** 2 + (Wm_gy - alphaWk_gy) ** 2).reshape(-1)))

            phi_kA = diags(
                ((Func_Phi_deriv(((alpha * Wk[i] + (1 - alpha) * Ik[i] - J[i]) ** 2))) * ((W - Ik[i]) ** 2)).reshape(
                    -1))
            phi_kB = (((Func_Phi_deriv(((alpha * Wk[i] + (1 - alpha) * Ik[i] - J[i]) ** 2))) * (W - Ik[i]) * (
                    J[i] - Ik[i])).reshape(-1))

            phi_alpha = diags(Func_Phi_deriv(alpha_gx ** 2 + alpha_gy ** 2).reshape(-1))
            L_alpha = sobelx.T.dot(phi_alpha.dot(sobelx)) + sobely.T.dot(phi_alpha.dot(sobely))

            L_f = sobelx.T.dot(phi_f).dot(sobelx) + sobely.T.dot(phi_f).dot(sobely)
            A_tilde_f = W_diag.T.dot(L_f).dot(W_diag)
            # Ax = b, setting up A
            if i == 0:
                A1 = phi_kA + lambda_a * L_alpha + beta * A_tilde_f
                b1 = phi_kB + beta * W_diag.dot(L_f).dot(W_m.reshape(-1))
            else:
                A1 += (phi_kA + lambda_a * L_alpha + beta * A_tilde_f)
                b1 += (phi_kB + beta * W_diag.T.dot(L_f).dot(W_m.reshape(-1)))

        alpha = linalg.spsolve(A1, b1).reshape(m, n, p)

        # plt.imshow(to_plot_normalize_image(alpha))
        # plt.draw()
        # plt.pause(0.001)

    return Wk, Ik, W, alpha


# TODO: Consider wrap around of indices to remove the edge at the end of sobel
# get Sobel sparse matrix for Y
def get_ySobel_matrix(m, n, p):
    size = m * n * p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_ysobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))


# get Sobel sparse matrix for X
def get_xSobel_matrix(m, n, p):
    size = m * n * p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_xsobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))


# get sobel coordinates for y
def _get_ysobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
        (i - 1, j, k, -2), (i - 1, j - 1, k, -1), (i - 1, j + 1, k, -1),
        (i + 1, j, k, 2), (i + 1, j - 1, k, 1), (i + 1, j + 1, k, 1)
    ]


# get sobel coordinates for x
def _get_xsobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
        (i, j - 1, k, -2), (i - 1, j - 1, k, -1), (i - 1, j + 1, k, -1),
        (i, j + 1, k, 2), (i + 1, j - 1, k, 1), (i + 1, j + 1, k, 1)
    ]


# Change to ravel index
# also filters the wrong guys
def _change_to_ravel_index(li, shape):
    li = filter(lambda x: _filter_list_item(x, shape), li)
    i, j, k, v = zip(*li)
    return zip(np.ravel_multi_index((i, j, k), shape), v)


def _filter_list_item(coord, shape):
    i, j, k, v = coord
    m, n, p = shape
    if i >= 0 and i < m and j >= 0 and j < n:
        return True


def Func_Phi_deriv(X, epsilon=1e-3):
    return 0.5 / Func_Phi(X, epsilon)


def Func_Phi(X, epsilon=1e-3):
    return np.sqrt(X + epsilon ** 2)


def changeContrastImage(J, I):
    cJ1 = J[0, 0, :]
    cJ2 = J[-1, -1, :]

    cI1 = I[0, 0, :]
    cI2 = I[-1, -1, :]

    I_m = cJ1 + (I - cI1) / (cI2 - cI1) * (cJ2 - cJ1)
    return I_m
