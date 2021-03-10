
import os
from PIL import Image
from shutil import copyfile, move


def filterImages(path=None):
    imgs = []
    filteredImages = []
    valid_images = [".jpg", ".png", ".jpeg", ".jfif"]

    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            imgs.append(f)

    print("Loaded Images.")
    print("Sorting {} images ...".format(len(imgs)))
    if not os.path.exists('{}./horizontal'.format(path)):
        os.mkdir('{}./horizontal'.format(path))

    if not os.path.exists('{}./veritcal'.format(path)):
        os.mkdir('{}./veritcal'.format(path))

    count = 0
    for i in imgs:
        image = Image.open(os.path.join(path, i))

        width, height = image.size

        image.close()

        if height / width > 1:
            move(os.path.join(path, i),
                 os.path.join(path + '/veritcal', i))
        else:
            move(os.path.join(path, i),
                 os.path.join(path + '/horizontal', i))

        count +=1
        if count % 100 == 0:
            print("{} of {} images".format(count, len(imgs)))

    print("---- Complete ----")


if __name__ == '__main__':
    folder_path = r""
    filterImages(folder_path)
