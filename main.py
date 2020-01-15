from itertools import islice
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
wm_type = 'ci'
source = 'cian4'
dir_images = f'./dataset/{source}'
res_dir = f'./dataset/{source}_' + str(''.join(rnd.choice('qwertyuiopasdfghjkl') for i in range(4)))  # + '/cropped'
files_number = 25
image_size = 1280

if not os.path.isdir(dir_images):
    os.mkdir(dir_images)

files = os.listdir(dir_images)

if len(files) == 0:
    photo_scrape(dir_images, files_number, wm_type)
else:
    print("All files downloaded")

# ----------------------------------------------------------------------------------------------------------------------

# Thresholds -----------------------------------------------------------------------------------------------------------
wm_detector_wm_thr = 0.05
wm_detector_img_thr = 0.1
wm_detector_tr_thr = 0.02
wm_crop_trh = 0.1

# INITIAL WATERMARK ESTIMATE & DETECTION -------------------------------------------------------------------------------
# get watermark gradient with median of images set TODO: make iterations possible

J = read_images(dir_images)

for i in range(1):
    images = preprocess(J, image_size)
    # images.update(preprocess(J, image_size, 'constant'))

    Wm_x, Wm_y, num_images = estimate_watermark(images)

    # crop watermark area and get cropped gradient
    cropped_Wm_x, cropped_Wm_y = crop_watermark_ignore_borders(Wm_x, Wm_y,
                                                               threshold=wm_crop_trh)

    # reconstruct watermark image with poisson
    W_m = poisson_reconstruct2(cropped_Wm_x, cropped_Wm_y)

    # # detect watermark on random photo
    # img_sample = rnd.choice(images)
    # # img_sample = cv2.imread(os.path.join(dir_images, img_name))
    # img_marked, wm_start, wm_end = watermark_detector(img_sample, cropped_Wm_x, cropped_Wm_y)

    J = get_cropped_images(J, cropped_Wm_x, cropped_Wm_y,
                           wm_thr=wm_detector_wm_thr,
                           img_thr=wm_detector_img_thr,
                           trash_thr=wm_detector_tr_thr,
                           )

    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

    for f, im in J.items():
        cv2.imwrite(
            (os.sep.join([os.path.abspath(res_dir), 'it_' + str(i) + '_' + f])),
            im
        )

    cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'it_' + str(i) + '_' + 'watermark.jpg'])), W_m)
    i += 1

    # plot_images([W_m])

# We are done with watermark estimation
# W_m is the cropped watermark
# ----------------------------------------------------------------------------------------------------------------------

# MULTI-IMAGE MATTING & RECONSTRUCTION ---------------------------------------------------------------------------------
# Wm = W_m - (255*to_plot_normalize_image(W_m))
# Wm = W_m - W_m.min()
Wm = W_m.copy()

# get threshold of W_m for alpha matte estimate
alpha_n = estimate_normalized_alpha(J, Wm, adaptive=True)
alpha_n = np.stack([alpha_n, alpha_n, alpha_n], axis=2)

C, est_Ik = estimate_blend_factor(J, Wm, alpha_n)
alpha = np.stack([C[i] * alpha_n[:, :, i] for i in [0, 1, 2]], axis=-1)

Wm = Wm + alpha * est_Ik
W = np.stack([Wm[:, :, i] / C[i] for i in [0, 1, 2]], axis=-1)

cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'watermark_0.jpg'])), W_m)
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'watermark.jpg'])), W)

# ----------------------------------------------------------------------------------------------------------------------

# Wm, alpha_n, cropped_Wm_x, cropped_Wm_y, est_Ik, Wm_x, Wm_y, images_raw, img_marked = \
#     None, None, None, None, None, None, None, None, None

# now we have the values of alpha, Wm, J
# Jk = np.array([next(iter(J.values()))])
Jk = np.array(list(islice(J.values(), 5)))
Wk, Ik, W, alpha1 = solve_images(Jk, W_m, alpha, W)
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'Wk.jpg'])), Wk)
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'Ik.jpg'])), Ik[0])
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'W.jpg'])), W)
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'alpha.jpg'])), alpha1)
plot_images([Ik[0]])

# W_m_threshold = (255*to_plot_normalize_image(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)
