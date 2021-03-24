# Script for plotting quicklooks radar/lidar wind products 

import sys
import datetime
import glob
import os
from pathlib import Path
from plot_routine import plotRoutine

################# change here! #################

days_range = (0, 2)
storagePath = "/work/marcus_mueller/routine/"

################################################

for dd in range(days_range[0], days_range[1]):
    now = datetime.datetime.now()
    end = datetime.datetime(now.year, now.month, now.day,23,59)-datetime.timedelta(days=dd)
    begin = datetime.datetime(now.year, now.month, now.day)-datetime.timedelta(days=dd)
    Path(storagePath+begin.strftime("%Y/%m/%d/")).mkdir(parents=True, exist_ok=True)
    plotRoutine(storagePath,begin, end)