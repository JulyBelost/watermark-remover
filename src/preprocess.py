import os
import sys
import cv2
import numpy as np


def preprocess(raw_dir, dest_dir, size):
	files = os.listdir(raw_dir)

	for file in files:
		path = (os.sep.join([os.path.abspath(raw_dir), file]))
		img = cv2.imread(path)
		if img is not None:
			m, n, p = img.shape
			m_t, n_t = (size-m)//2, (size-n)//2
			final_img = np.pad(img, ((m_t, size-m-m_t), (n_t, size-n-n_t), (0, 0)), mode='constant')
			cv2.imwrite(os.sep.join([dest_dir, file]), final_img)
			print("Saved to : %s"%(file))


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Format : %s <foldername>"%(sys.argv[0]))
	else:
		preprocess(sys.argv[1])