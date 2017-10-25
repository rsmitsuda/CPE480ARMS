from PIL import Image
from deap import base, creator, tools
import sys
import os
import io
import random

NUM_GEN = 50
PROB_MATE = 0.5
PROB_MUT = 0.4

# Wrapper to extract bytes from an image
class ImageWrapper(object):
    def __init__(self, filename):
        img = Image.open(filename)
        bytesObj = io.BytesIO()

        img.save(bytesObj, img.format)
        img.close()

        self.bytes = bytearray(bytesObj.getvalue())

        bytesObj.close()

    def copy(self, img):
        img.bytes = bytearray(self.bytes)

# Creates a DEAP bytearray individual from a filename
def createImageInd(filename):
    img = creator.Image()   
    imgWrap = ImageWrapper(filename)
    img[:] = imgWrap.bytes

    return img

def evaluate(img):
    return float(sum(img)), (len(img))

def validateArgs(args):
    parsed = []

    for i in range(0, len(args), 2):
        try:
            inputPair = (args[i], float(args[i + 1]))
        except ValueError:
            sys.stderr.write('Invalid weight - %s\n' % (args[i + 1]))
            sys.exit(1)

        parsed.append(inputPair)

    return parsed

def main():
    if len(sys.argv) < 3:
        print 'Usage: platform <img1> <weight1> ...'
        sys.exit(1)

    args = validateArgs(sys.argv[1:])

    creator.create('MaxFitness', base.Fitness, weights=(1.0,))
    creator.create('Image', bytearray, fitness=creator.MaxFitness)

    toolbox = base.Toolbox()
    toolbox.register('addImg', createImageInd)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=1,\
            indpb=0.01)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('evaluate', evaluate)

    runIterations(toolbox, args)

def runIterations(toolbox, args):
    population = []
       
    for a in args:
        try:
            population.append(toolbox.addImg(a[0]))
        except Exception as e:
            print e
            sys.stderr.write('Invalid filename - %s\n' % (a[0]))
            sys.exit(1)

    for p in population:
        p.fitness.values = evaluate(p)

    for i in range(NUM_GEN):
        best = toolbox.select(population, len(population))
        best = [toolbox.clone(p) for p in best]

        for j in range(0, len(best), 2):
            img1 = best[i]

            if (i < len(best) - 1):
                img2 = best[i + 1]
                
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

    firstImage = Image.frombytes('RGB', len(best[0]), best[0])
    firstImage.show()
    firstImage.save('output', 'PNG')

if __name__ == '__main__':
    main()
