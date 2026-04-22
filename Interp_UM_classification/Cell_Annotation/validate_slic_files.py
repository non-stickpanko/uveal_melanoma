import os
import numpy as np
from skimage.io import imread


def validate_slic_folder(image_folder):
    valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    if not os.path.isdir(image_folder):
        print(f"ERROR: image folder not found: {image_folder}")
        return

    files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(valid_ext)])
    print(f"Found {len(files)} image file(s) in: {image_folder}")

    if not files:
        return

    ok_count = 0
    fail_count = 0

    for fname in files:
        image_path = os.path.join(image_folder, fname)
        base = os.path.splitext(fname)[0]
        npy_path = os.path.join(image_folder, base + ".npy")

        print("-" * 60)
        print(f"IMAGE: {fname}")

        if not os.path.exists(npy_path):
            print(f"  FAIL: missing npy file -> {base}.npy")
            fail_count += 1
            continue

        try:
            image = imread(image_path)
            slic = np.load(npy_path)

            print(f"  image shape: {image.shape}, dtype: {image.dtype}")
            print(f"  slic  shape: {slic.shape}, dtype: {slic.dtype}")

            passed = True

            if getattr(slic, 'ndim', None) != 2:
                print("  FAIL: SLIC array is not 2D")
                passed = False

            img_h, img_w = image.shape[:2]
            if slic.shape != (img_h, img_w):
                print(f"  FAIL: shape mismatch, expected {(img_h, img_w)}")
                passed = False

            if not np.issubdtype(slic.dtype, np.integer):
                print("  FAIL: SLIC dtype is not integer")
                passed = False

            uniq = np.unique(slic)
            print(f"  unique labels: {len(uniq)}")
            if len(uniq) > 0:
                print(f"  min label: {uniq.min()}, max label: {uniq.max()}")

            if len(uniq) == 0:
                print("  FAIL: no labels found")
                passed = False

            if passed:
                print("  PASS: file matches SuperPixelAnno expected format")
                ok_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"  FAIL: could not load/validate -> {e}")
            fail_count += 1

    print("=" * 60)
    print(f"Validation complete. PASS: {ok_count}, FAIL: {fail_count}")


if __name__ == "__main__":
    image_folder = r"C:\Users\diavi\OneDrive\Desktop\PINN_project\uveal_melanoma\Interp_UM_classification\images\slide_images"
    validate_slic_folder(image_folder)