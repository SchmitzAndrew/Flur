from PIL import Image, ImageFilter
import numpy


def blur(img, part, intensity):
    im = Image.open(img)
    for x, y, w, h in part:
        box = (x, y, x + w, y + h)
        region = im.crop(box)
        blurred = region.filter(ImageFilter.GaussianBlur(intensity))
        im.paste(blurred, box)
    blurred_image = im.save("out/blurredimage.jpg")

