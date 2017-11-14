from PIL import Image, ImageFile, ImageStat, ImageFilter, ImageChops
import io
import numpy
import math

PPM_HEADER = 3

# Wrapper to extract bytes from an image
class ImageWrapper(object):
    def __init__(self, filename, weight):
        img = Image.open(filename).convert('RGB')
        self.trim(img)
        self.filename = filename

        bytesObj = io.BytesIO()

        self.width = img.width
        self.height = img.height
        self.weight = weight

        img.save(bytesObj, 'PPM')
        img.close()

        bytesObj.seek(0)
        self.rawBytes = bytesObj.readlines()
        self.header = self.rawBytes[:PPM_HEADER]
        self.bytes = bytearray(self.rawBytes[PPM_HEADER])

        bytesObj.close()

    def trim(self, img):
        bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()

        if bbox:
            return img.crop(bbox)

        raise ValueError('missing bounding box for image %s' % self.filename)

def fitnessFunction(image):
  #Image Info (w/o background color)
  width = image.width
  height = image.height

  # [(count, (R, G, B)), ...]
  colors = image.getcolors()
  pixels = list(image.getdata())

  #Color Info (from each band)
  averageColor = ImageStat.Stat(image).mean
  minColor = tuple(pixel[0] for pixel in ImageStat.Stat(image).extrema)
  maxColor = tuple(pixel[1] for pixel in ImageStat.Stat(image).extrema)
  medianColor = ImageStat.Stat(image).median
  stddev = ImageStat.Stat(image).stddev

  #squares the value of each band, adds them together, and takes the sqrt of the value
  medianVal = math.sqrt((medianColor[0]**2) + (medianColor[1]**2) + (medianColor[2]**2))
  averageVal = math.sqrt((averageColor[0]**2) + (averageColor[1]**2) + (averageColor[2]**2))
  stddevVal = math.sqrt((stddev[0]**2) + (stddev[1]**2) + (stddev[2]**2))

  fitnessScore = 0

  # chi square analysis
  # if we want/like more variety in colors (lower colorVarianceScore) then
  # the farther away the average is from median, the worse the fitness score since median is always middle of the results
  # less color variety = higher colorVarianceScore
  colorVarianceScore = (1/((medianVal - averageVal)**2/averageVal))*10
  # print colorVarianceScore

  # if width > height -> AR > 1
  # if height > width -> AR < 1
  # if width = height -> AR = 1
  aspectRatio = (float(width)/float(height))

  # higher sizeScore = larger image
  # lowewr sizeScore = smaller image
  imageArea = float(width) * float(height)
  sizeScore = math.sqrt(imageArea)*0.1
  # print sizeScore

  # if we want large images, add sizescore
  # if we want small images, subtract sizeScore
  # if we dont care about size, ignore sizeScore

  # if we want more colors, low colorVarianceScore is good
  # if we want less colors, high colorVarianceScore is good

  # large image, high color variance
  fitnessScore = colorVarianceScore + sizeScore

  # print width, height, aspectRatio, sizeScore

if __name__ == '__main__':
    main()
