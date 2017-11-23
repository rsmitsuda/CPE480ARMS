from ImageWrapper import *
from deap import base, creator, tools
import sys
import os
import io
import random
import numpy

NUM_IND = 30
NUM_GEN = 50
PROB_MATE = 0.6
PROB_MUT = 0.5
RGB = 3
W_AVG = 0.3
W_MODE = 0.1
W_STD = 0.3
W_DISTINCT = 0.5

# Clamps a value in the range of 0 to 255
def clamp(val):
    if val < 0.0:
        val = 0.0
    elif val > 255.0:
        val = 255.0

    return int(val)

# Creates a DEAP bytearray individual from a filename
def createImageInd(filename, weight):
    img = creator.Image()   
    imgWrap = ImageWrapper(filename, weight)
    img[:] = imgWrap.bytes
    img.width = imgWrap.width
    img.height = imgWrap.height
    img.header = imgWrap.header
    img.weight = imgWrap.weight

    return img

# Blends two images based on their weight
def blendImgs(img1, img2):
    tools.cxTwoPoint(img1, img2)

    img1.weight = img2.weight = (img1.weight + img2.weight) / 2

# Simple hash function for rgb color values
def simpleHash(r, g, b):
    return (int(r) << 16) + (int(g) << 8) + int(b)

def evaluate(img):
    histogram = {}
    curR, curG, curB = -1, -1, -1
    numColors = 0

    for i in range(0, len(img), RGB):
        r, g, b = img[i], img[i + 1], img[i + 2]
        color = simpleHash(r, g, b)

        # Count the number of contiguous colors
        if curR != r or curG != g or curB != b:
            curR = r
            curG = g
            curB = b

            numColors += 1

        if color not in histogram:
            histogram[color] = 0

        histogram[color] += 1

    avgColor = numpy.mean(img) / 255.0
    modeColor = max([(color, histogram[color]) for color in histogram], \
        key=lambda s : s[1])[0]

    modeVal = float(modeColor) / 0xffffff

    stdDev = numpy.std(img) / 255.0

    total = img.weight * (W_AVG * avgColor + W_MODE * modeVal - W_STD * stdDev)\
       / (W_DISTINCT * numColors);

    return (total,)

def validateArgs(args):
    parsed = []

    if len(args) % 2:
        sys.stderr.write('invalid number of arguments\n')
        sys.exit(1)

    for i in range(0, len(args), 2):
        try:
            inputPair = (args[i], float(args[i + 1]))

            if inputPair[1] < 0.0 or inputPair[1] > 1.0:
                raise ValueError('invalid weight')

        except ValueError:
            sys.stderr.write('Invalid weight - %s\n' % (args[i + 1]))
            sys.exit(1)

        parsed.append(inputPair)

    if sum([a[1] for a in parsed]) != 1.0:
        sys.stderr.write('Weights must add up to 1\n')
        sys.exit(1)

    return parsed

def main():
    if len(sys.argv) < 3:
        print 'Usage: platform <img1> <weight1> ...'
        sys.exit(1)

    args = validateArgs(sys.argv[1:])

    creator.create('MaxFitness', base.Fitness, weights=(1.0,))
    creator.create('Image', list, fitness=creator.MaxFitness, width=0, \
            height=0, header=None, weight=0.0)

    toolbox = base.Toolbox()
    toolbox.register('addImg', createImageInd)
    toolbox.register('mate', blendImgs)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.01)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('evaluate', evaluate)

    runIterations(toolbox, args)

def runIterations(toolbox, args):
    population = []
    i = 0

    # Make enough individuals for a population, with possible duplicates
    while i < NUM_IND:
        a = args[i % len(args)]

        try:
            population.append(toolbox.addImg(a[0], a[1]))
        except Exception as e:
            print e
            sys.stderr.write('Invalid filename - %s\n' % (a[0]))
            sys.exit(1)

        i += 1

    for p in population:
        p.fitness.values = evaluate(p)

    for i in range(NUM_GEN):
        best = toolbox.select(population, len(population))
        best = [toolbox.clone(p) for p in best]

        for j in range(0, len(best), 2):
            img1 = best[j]

            if (j < len(best) - 1):
                img2 = best[j + 1]
                
                if random.random() < PROB_MATE:
                    toolbox.mate(img1, img2)
                    del img1.fitness.values
                    del img2.fitness.values

        for img in best:
            if random.random() < PROB_MUT:
                toolbox.mutate(img)

                del img.fitness.values

        for img in best:
            if not img.fitness.valid:
                img.fitness.values = toolbox.evaluate(img)

        population[:] = best

    outputImages(population, 1)

def outputImages(population, n):
    best = tools.selBest(population, n)   

    bestImage = best[0]

    numVals = [int(b) for b in bestImage]

    for i in range(len(numVals)):
        if numVals[i] > 255:
            numVals[i] = 255
        if numVals[i] < 0:
            numVals[i] = 0

    bestBuffer = ''.join([chr(i) for i in numVals])

    with open('output.ppm', 'wb+') as f:
        for l in bestImage.header:
            f.write(l)

        f.write(bestBuffer)

    timg = Image.open('output.ppm')
    # print("sadf")
    # timg.show()
    timg.save('output.png', 'PNG')

if __name__ == '__main__':
    main()
