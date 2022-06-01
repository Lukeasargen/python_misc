# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
import os
import glob
import time

import torch
import clip
from PIL import Image
from shutil import move


def main():
    num_images = 1000
    root = r"folder path"
    labels = [
        "dog",
        "cat",
        "bird",
        "plane",
    ]
    model_name = "ViT-B/16"

    # Create folders for labels
    class_folder_paths = []  # Absolute path to destination folder
    for cat in labels:
        cat_path = os.path.join(root, cat)
        class_folder_paths.append(cat_path)
        if not os.path.exists(cat_path):
            os.mkdir(cat_path)

    # Get all the images
    image_types = ["*.jpg", "*.png", "*.jpeg"]
    images = [f for ext in image_types for f in glob.glob(os.path.join(root, ext))]
    print("{} total images.".format(len(images)))
    max_count = min(num_images, len(images))
    print(" * Sorting {} ...".format(max_count))
    counts = [0]*len(labels)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(clip.available_models())
    model, preprocess = clip.load(model_name, device=device, jit=False)
    text = clip.tokenize(labels).to(device)

    start_time = time.time()
    for i in range(max_count):
        image = preprocess(Image.open(images[i])).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
        class_prob, class_num = torch.max(logits_per_image.squeeze(0), dim=0)
        counts[int(class_num)] += 1
        try:
            move(images[i], class_folder_paths[int(class_num)])
        except:
            print("Failed to move {}".format(images[i]))

        if (i+1) % 50 == 0:
            count = i+1
            t2 = time.time() - start_time
            rate = count/t2
            est = t2/count * (max_count-count)
            print("{}/{} images. {:.2f} seconds. {:.2f} images per seconds. {:.2f} seconds remaining.".format(count, max_count, t2, rate, est))
    duration = time.time() - start_time
    print(" * Sort Complete")
    print(" * Duration {:.2f} Seconds".format(duration))
    print(" * {:.2f} Images per Second".format(max_count/duration))

    lw = len(max(labels, key=lambda x: len(x)))
    print(f'{"Label":>{lw}s}: {"Count":>6} {"Perc":>6}')
    for l, p in zip(labels, counts):
        print(f'{l:>{lw}s}: {p:>6.0f} {100*p/sum(counts):>6.2f}')


if __name__ == "__main__":
    main()