import cv2
import os
import random as rnd
from src.estimate_watermark import *
from src.preprocess import *
from src.image_getter import *
from src.watermark_reconstruct import *

# GET IMAGES ----------------------------------------------------------
# Download random images from database
# and prepare them for the main process(make same shape)
# if there is no any yet
#
# folder name of preprocessed images with watermarks
dir_images = './dataset/source'
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

# -------------------------------------------------------------------------

# INITIAL WATERMARK DETECTION ---------------------------------------------
# get watermark gradient with median of images set TODO: make iterations possible
images_raw = read_images(dir_images)

for i in range(2):
    images = preprocess(images_raw, image_size)

    Wm_x, Wm_y, num_images = estimate_watermark(images)

    # crop watermark area and get cropped gradient
    cropped_Wm_x, cropped_Wm_y = crop_watermark(Wm_x, Wm_y)

    # reconstruct watermark image with poisson
    W_m = poisson_reconstruct2(cropped_Wm_x, cropped_Wm_y)

    # # detect watermark on random photo
    # img_sample = rnd.choice(images)
    # # img_sample = cv2.imread(os.path.join(dir_images, img_name))
    # img_marked, wm_start, wm_end = watermark_detector(img_sample, cropped_Wm_x, cropped_Wm_y)

    images_raw = {}
    for file, img in images.items():
        img_marked, wm_start, wm_end = watermark_detector(img, cropped_Wm_x, cropped_Wm_y)
        img_result = img[max(wm_start[0], 0):wm_start[0] + wm_end[0], max(wm_start[1], 0):wm_start[1] + wm_end[1], :]

        if not img_result.any():
            continue

        cv2.imwrite((os.sep.join([os.path.abspath(cropped_wm_dir), 'it_' + str(i) + '_' + file])), img_result)

        images_raw[file] = img_result

    cv2.imwrite((os.sep.join([os.path.abspath(cropped_wm_dir), 'it_' + str(i) + '_' + 'watermark.jpg'])), W_m)
    i += 1

    # # plotting images
    # images_for_plotting = [img_marked, W_m]
    #
    # for img in images_for_plotting:
    #     img_res = img[:, :, ::-1]
    #     plt.figure(dpi=600)
    #     plt.imshow(img_res)
    #     plt.xticks([]), plt.yticks([])
    #     plt.show()
    # # -------------------------------------------------------------------------

# We are done with watermark estimation
# W_m is the cropped watermark
# -------------------------------------------------------------------------

#  ---------------------------------------------

J, img_paths = get_cropped_images(
    'images/fotolia_processed', num_images, wm_start, wm_end, cropped_Wm_x.shape)
# get a random subset of J
idx = [389, 144, 147, 468, 423, 92, 3, 354, 196, 53, 470, 445, 314, 349, 105, 366, 56, 168, 351, 15, 465, 368, 90, 96,
       202, 54, 295, 137, 17, 79, 214, 413, 454, 305, 187, 4, 458, 330, 290, 73, 220, 118, 125, 180, 247, 243, 257, 194,
       117, 320, 104, 252, 87, 95, 228, 324, 271, 398, 334, 148, 425, 190, 78, 151, 34, 310, 122, 376, 102, 260]
idx = idx[:25]
# for img in J:
#     img_res = img[:, :, ::-1]
#     plt.figure(dpi=600)
#     plt.imshow(img_res)
#     plt.xticks([]), plt.yticks([])
#     plt.show()
# Wm = (255*to_plot_normalize_image(W_m))
Wm = W_m - W_m.min()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm, num_images)
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
C, est_Ik = estimate_blend_factor(J, Wm, alph)

alpha = alph.copy()
for i in range(3):
    alpha[:, :, i] = C[i] * alpha[:, :, i]

Wm = Wm + alpha * est_Ik

img_res = Wm[:, :, ::-1]
plt.figure(dpi=600)
plt.imshow(Wm)
plt.xticks([]), plt.yticks([])
plt.show()

W = Wm.copy()
for i in range(3):
    W[:, :, i] /= C[i]

Jt = J[:25]
Wm, J, alph, alph_est, cropped_Wm_x, cropped_Wm_y, est_Ik, Wm_x, Wm_y, img_sample, img_marked = \
    None, None, None, None, None, None, None, None, None, None, None

# now we have the values of alpha, Wm, J
# Solve for all images
print(type(Jt))
print(type(W_m))
print(type(alpha))
print(type(W))
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
# W_m_threshold = (255*to_plot_normalize_image(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)
