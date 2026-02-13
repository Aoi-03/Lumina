from astropy.io import fits
import cv2
import numpy as np

# open the FITS file
with fits.open("data/2636m273_ac51-w2-int-3.fits") as hdul:
    img_data = None
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        if data is not None:
            img_data = data
            break

if img_data is None:
    raise ValueError("No image data found in this FITS file!")

# convert to float32 and normalize
img_data = np.nan_to_num(np.array(img_data, dtype=np.float32))
dst = np.zeros_like(img_data)
norm = cv2.normalize(src=img_data, dst=dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
norm = np.uint8(norm)

# save the image
if norm is not None and isinstance(norm, np.ndarray):
    cv2.imwrite("data/output.png", norm)
    print("Saved image as output.png")
else:
    raise ValueError("Normalized image is not a valid numpy array.")



