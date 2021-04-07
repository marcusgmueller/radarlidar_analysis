# Script for plotting quicklooks radar/lidar wind products 

import sys
import datetime
import glob
import os
import matplotlib 
matplotlib.use('Agg') 
from pathlib import Path
from RadarLidarWindSpeed import RadarLidarWindSpeed


################# change here! #################

days_range = (0, 1)#change back to 2
storagePath = "/work/marcus_mueller/routine/"

################################################

for dd in range(days_range[0], days_range[1]):
    now = datetime.datetime.now()
    end = datetime.datetime(now.year, now.month, now.day,23,59)-datetime.timedelta(days=dd)
    begin = datetime.datetime(now.year, now.month, now.day)-datetime.timedelta(days=dd)
    Path(storagePath+begin.strftime("%Y/%m/%d/")).mkdir(parents=True, exist_ok=True)
    storagePath = storagePath+begin.strftime("%Y/%m/%d/")
    #plotRoutine(storagePath,begin, end)

    #new approach
    analysis = RadarLidarWindSpeed(begin, end)
    analysis.importDataset()
    analysis.calculateSpeedFusion()
    analysis.calculateDifferences()
    analysis.calculateDirectionFusion()
    analysis.calculateAvailability()
    analysis.windspeedFullHeightCoveragePlot(storagePath)
    analysis.windspeedBoundaryLayerCoveragePlot(storagePath)
    analysis.windspeedFullHeightOverviewPlot(storagePath)
    analysis.windspeedBoundaryLayerOverviewPlot(storagePath)
    analysis.winddirectionFullHeightOverviewPlot(storagePath)
    analysis.winddirectionBoundaryLayerOverviewPlot(storagePath)
    analysis.exportNCDF(storagePath)
    