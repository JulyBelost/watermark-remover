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
dir_images = './images_dataset/img_prepared'
dir_images_raw = './images_dataset/img_raw'
files_number = 25
image_size = 1280

if not os.path.isdir(dir_images):
    os.mkdir(dir_images)

files = os.listdir(dir_images)

if len(files) == 0:
    if not os.path.isdir(dir_images_raw):
        os.mkdir(dir_images_raw)

    files_raw = os.listdir(dir_images_raw)

    if len(files_raw) == 0:
        photo_scrape(dir_images_raw, files_number)
    else:
        print("All files downloaded")

    preprocess(dir_images_raw, dir_images, image_size)
else:
    print("All files prepared")
# -------------------------------------------------------------------------

# INITIAL WATERMARK DETECTION ---------------------------------------------
Wm_x, Wm_y, num_images = estimate_watermark(dir_images)
cropped_Wm_x, cropped_Wm_y = crop_watermark(Wm_x, Wm_y)
W_m = poisson_reconstruct(cropped_Wm_x, cropped_Wm_y)
# est = poisson_reconstruct(Wm_x, Wm_y, np.zeros(Wm_x.shape)[:,:,0])
img_name = rnd.choice(os.listdir(dir_images))
sample = os.path.join(dir_images, img_name)
# random photo
img = cv2.imread(sample)
im, start, end = watermark_detector(img, cropped_Wm_x, cropped_Wm_y)
im_size = Wm_x.nbytes

plt.imshow(im)
plt.show()
plt.imshow(W_m)
plt.show()
# We are done with watermark estimation
# W_m is the cropped watermark

J, img_paths = get_cropped_images(
    'images/fotolia_processed', num_images, start, end, cropped_Wm_x.shape)
# get a random subset of J
idx = [389, 144, 147, 468, 423, 92, 3, 354, 196, 53, 470, 445, 314, 349, 105, 366, 56, 168, 351, 15, 465, 368, 90, 96,
       202, 54, 295, 137, 17, 79, 214, 413, 454, 305, 187, 4, 458, 330, 290, 73, 220, 118, 125, 180, 247, 243, 257, 194,
       117, 320, 104, 252, 87, 95, 228, 324, 271, 398, 334, 148, 425, 190, 78, 151, 34, 310, 122, 376, 102, 260]
idx = idx[:25]
# Wm = (255*PlotImage(W_m))
Wm = W_m - W_m.min()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm, num_images)
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
C, est_Ik = estimate_blend_factor(J, Wm, alph)

alpha = alph.copy()
for i in range(3):
    alpha[:, :, i] = C[i] * alpha[:, :, i]

Wm = Wm + alpha * est_Ik
plt.imshow(Wm)
plt.show()

W = Wm.copy()
for i in range(3):
    W[:, :, i] /= C[i]

Jt = J[:25]
Wm = None
J = None
alph = None
alph_est = None
cropped_Wm_x = None
cropped_Wm_y = None
est_Ik = None
Wm_x = None
Wm_y = None
im = None
img = None

# now we have the values of alpha, Wm, J
# Solve for all images
print(type(Jt))
print(type(W_m))
print(type(alpha))
print(type(W))
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
# W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)
