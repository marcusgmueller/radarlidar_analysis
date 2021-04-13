# Script for plotting quicklooks radar/lidar wind products 

import sys
import datetime
import glob
import os
from pathlib import Path
import matplotlib 
matplotlib.use('Agg') 
from RadarLidarWindSpeed import RadarLidarWindSpeed
from dateutil.relativedelta import *
from datetime import datetime

################# change here! #################

storagePath = "/work/marcus_mueller/history/"

################################################


start = datetime(2019, 4, 1)
for i in range(23): # n months from start
    dateBegin=start+relativedelta(months=+i)
    print(dateBegin)
    dateEnd=start+relativedelta(months=+i+1)
    analysis = RadarLidarWindSpeed(dateBegin, dateEnd)
    analysis.importDataset()
    #analysis.calculateSpeedFusion()
    #analysis.calculateDifferences()
    #analysis.calculateDirectionFusion()
    analysis.calculateAvailability()


    analysis.availabilityPlot(storagePath)
    analysis.windspeedFullHeightCoveragePlot(storagePath)
    analysis.windspeedBoundaryLayerCoveragePlot(storagePath)