from PIL import Image, ImageFile, ImageStat, ImageFilter, ImageChops
import io
import numpy

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def main():
   imagePath = "bird.png"
   ImageFile.LOAD_TRUNCATED_IMAGES = True

   image = Image.open(imagePath).convert("RGB")
   bytesObj = io.BytesIO()

   trimmedImage = trim(image)

   #Image Info (w/o background color)
   width = trimmedImage.width
   height = trimmedImage.height

   # [(count, (R, G, B)), ...]
   colors = image.getcolors()
   pixels = list(image.getdata())

   #Color Info (from each band)
   averageColor = ImageStat.Stat(image).mean
   minColor = tuple(pixel[0] for pixel in ImageStat.Stat(image).extrema)
   maxColor = tuple(pixel[1] for pixel in ImageStat.Stat(image).extrema)
   medianColor = ImageStat.Stat(image).median
   stddev = ImageStat.Stat(image).stddev

   # image.save("parsed.png", format="PNG")
   # out = image.filter(ImageFilter.FIND_EDGES)
   trimmedImage.save("out.png", format="PNG")



if __name__ == '__main__':
    main()