import os
import numpy as np
from skimage.io import imread
from skimage.segmentation import slic
from skimage.util import img_as_float


def generate_slic_for_folder(
    image_folder,
    n_segments=400,
    compactness=5,
    sigma=1,
    start_label=0
):
    valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    if not os.path.isdir(image_folder):
        print(f"Image folder not found: {image_folder}")
        return

    files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(valid_ext)])

    if not files:
        print(f"No image files found in: {image_folder}")
        return

    print(f"Found {len(files)} image(s) in: {image_folder}")

    for fname in files:
        image_path = os.path.join(image_folder, fname)
        base_name = os.path.splitext(fname)[0]
        out_path = os.path.join(image_folder, base_name + ".npy")

        try:
            image = imread(image_path)
            print(f"Processing: {fname} | shape={image.shape} | dtype={image.dtype}")

            if image.ndim != 3:
                print(f"Skipping {fname}: expected RGB image, got shape {image.shape}")
                continue

            if image.shape[2] == 4:
                image = image[:, :, :3]

            if image.shape[2] != 3:
                print(f"Skipping {fname}: expected 3 channels, got {image.shape[2]}")
                continue

            image_for_slic = img_as_float(image)

            segments = slic(
                image_for_slic,
                n_segments=n_segments,
                compactness=compactness,
                sigma=sigma,
                start_label=start_label,
                channel_axis=-1
            )

            np.save(out_path, segments.astype(np.int32))
            print(f"Saved: {out_path} | seg_shape={segments.shape} | labels={len(np.unique(segments))}")

        except Exception as e:
            print(f"Failed on {fname}: {e}")


if __name__ == "__main__":
    image_folder = r"C:\Users\diavi\OneDrive\Desktop\PINN_project\uveal_melanoma\Interp_UM_classification\images\slide_images"

    generate_slic_for_folder(
        image_folder=image_folder,
        n_segments=400,
        compactness=5,
        sigma=1,
        start_label=0
    )