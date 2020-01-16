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
source = 'cian'
dir_images = f'./dataset/{source}'
res_dir = f'./dataset/{source}_' + str(''.join(rnd.choice('qwertyuiopasdfghjkl') for i in range(4)))  # + '/cropped'
files_number = 25 if len(sys.argv) < 2 else int(sys.argv[1])
image_size = 1280

if not os.path.isdir(dir_images):
    os.mkdir(dir_images)

files = os.listdir(dir_images)

if len(files) == 0:
    photo_scrape(dir_images, files_number, wm_type)
else:
    print("All files downloaded")

# ----------------------------------------------------------------------------------------------------------------------

# INITIAL WATERMARK ESTIMATE & DETECTION -------------------------------------------------------------------------------
# Thresholds -----------------------------------------------------------------------------------------------------------
wm_crop_trh = 0.05
wm_detector_wm_thr = 0.05
wm_detector_img_thr = 0.1
wm_detector_tr_thr = 0.02
# ----------------------------------------------------------------------------------------------------------------------
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
# Thresholds -----------------------------------------------------------------------------------------------------------
alpha_thr = 170
alpha_adapt_thr = 21
# ----------------------------------------------------------------------------------------------------------------------
J = [im for im in J.values() if im.shape == W_m.shape][:4]
# Wm = W_m.copy()
images, cropped_Wm_x, cropped_Wm_y, Wm_x, Wm_y = None, None, None, None, None

# get threshold of W_m for alpha matte estimate
alpha_n = estimate_normalized_alpha(J, W_m, threshold=alpha_thr, adaptive_threshold=alpha_adapt_thr)
alpha_n = np.stack([alpha_n, alpha_n, alpha_n], axis=2)
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'alpha_n.jpg'])), alpha_n)

C, est_Ik = estimate_blend_factor(J, W_m, alpha_n)
alpha = np.stack([C[i] * alpha_n[:, :, i] for i in [0, 1, 2]], axis=-1)

W_m = W_m + alpha * est_Ik
W = np.stack([W_m[:, :, i] / C[i] for i in [0, 1, 2]], axis=-1)

cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'watermark_0.jpg'])), W_m)
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'watermark.jpg'])), W)

# ----------------------------------------------------------------------------------------------------------------------

# Wm, alpha_n, cropped_Wm_x, cropped_Wm_y, est_Ik, Wm_x, Wm_y, images_raw, img_marked = \
#     None, None, None, None, None, None, None, None, None
J = np.array(J)
# now we have the values of alpha, Wm, J
Wk, Ik, W, alpha1 = solve_images(J, W_m, alpha, W)
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'Wk.jpg'])), Wk)
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'Ik.jpg'])), Ik[0])
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'W.jpg'])), W)
cv2.imwrite((os.sep.join([os.path.abspath(res_dir), 'alpha.jpg'])), alpha1)
for i in range(Ik.shape[0]):
    cv2.imwrite(
        (os.sep.join([os.path.abspath(res_dir), 'Ik_' + str(i) + '.jpg'])),
        Ik[i, ...]
    )
plot_images([Ik[0]])
print('done')
# W_m_threshold = (255*to_plot_normalize_image(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)
