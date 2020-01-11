import cv2
import os
import random as rnd
from src.estimate_watermark import *
from src.preprocess import *
from src.image_getter import *
from src.watermark_reconstruct import *

# GET IMAGES -----------------------------------------------------------------------------------------------------------
# Download random images from database
# and prepare them for the main process(make same shape)
# if there is no any yet
#
# folder name of preprocessed images with watermarks
dir_images = './dataset/domofond'
cropped_wm_dir = './dataset/result_' + str(''.join(rnd.choice('qwertyuiopasdfghjkl') for i in range(5))) + '/cropped'
files_number = 25
image_size = 1280

if not os.path.isdir(dir_images):
    os.mkdir(dir_images)
if not os.path.isdir(cropped_wm_dir):
    os.makedirs(cropped_wm_dir)

files = os.listdir(dir_images)

if len(files) == 0:
    photo_scrape(dir_images, files_number)
else:
    print("All files downloaded")

# ----------------------------------------------------------------------------------------------------------------------

# INITIAL WATERMARK ESTIMATE & DETECTION -------------------------------------------------------------------------------
# get watermark gradient with median of images set TODO: make iterations possible
J = read_images(dir_images)

for i in range(1):
    images = preprocess(J, image_size)

    Wm_x, Wm_y, num_images = estimate_watermark(images)

    # crop watermark area and get cropped gradient
    cropped_Wm_x, cropped_Wm_y = crop_watermark(Wm_x, Wm_y)

    # reconstruct watermark image with poisson
    W_m = poisson_reconstruct2(cropped_Wm_x, cropped_Wm_y)

    # # detect watermark on random photo
    # img_sample = rnd.choice(images)
    # # img_sample = cv2.imread(os.path.join(dir_images, img_name))
    # img_marked, wm_start, wm_end = watermark_detector(img_sample, cropped_Wm_x, cropped_Wm_y)

    J = get_cropped_images(images, cropped_Wm_x, cropped_Wm_y)

    for f, im in J.items():
        cv2.imwrite(
            (os.sep.join([os.path.abspath(cropped_wm_dir), 'it_' + str(i) + '_' + f])),
            im
        )

    cv2.imwrite((os.sep.join([os.path.abspath(cropped_wm_dir), 'it_' + str(i) + '_' + 'watermark.jpg'])), W_m)
    i += 1

    # plot_images([img_marked, W_m])

# We are done with watermark estimation
# W_m is the cropped watermark
# ----------------------------------------------------------------------------------------------------------------------

# MULTI-IMAGE MATTING & RECONSTRUCTION ---------------------------------------------------------------------------------
# Wm = W_m - (255*to_plot_normalize_image(W_m))
# Wm = W_m - W_m.min()
Wm = W_m.copy()

# get threshold of W_m for alpha matte estimate
alpha_n_1d = estimate_normalized_alpha(J, Wm)
alpha_n = np.stack([alpha_n_1d, alpha_n_1d, alpha_n_1d], axis=2)
C, est_Ik = estimate_blend_factor(J, Wm, alpha_n)

alpha = np.zeros(alpha_n.shape)
for i in range(3):
    alpha[:, :, i] = C[i] * alpha_n[:, :, i]

Wm = Wm + alpha * est_Ik

W = Wm.copy()
for i in range(3):
    W[:, :, i] /= C[i]

# ----------------------------------------------------------------------------------------------------------------------

Wm, alpha_n, alph_est, cropped_Wm_x, cropped_Wm_y, est_Ik, Wm_x, Wm_y, images_raw, img_marked = \
    None, None, None, None, None, None, None, None, None, None

# now we have the values of alpha, Wm, J
# Solve for all images
Wk, Ik, W, alpha1 = solve_images(J[:1], W_m, alpha, W)
# W_m_threshold = (255*to_plot_normalize_image(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)
