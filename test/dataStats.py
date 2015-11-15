import time

from util.utils import computeDataStatistics


if __name__ == "__main__":
    start = time.time()
    dataSet = "dev"
    computeDataStatistics(dataSet=dataSet)
    print "Time to compute statistics for %s dataset: %f" \
        %(dataSet, (time.time() - start))