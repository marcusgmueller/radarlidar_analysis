from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy.ma as ma
from matplotlib.colors import ListedColormap
from datetime import datetime, timedelta
import matplotlib
import os
import math


class RadarLidarWindSpeed:
    # Variables
    dateBegin = datetime.timestamp(datetime.now())
    dateEnd = datetime.timestamp(datetime.now())
    days = []
    hours = []
    heightGrid = list(range(0,15000,50))#range(0,15012,36)
    dataframe = pd.DataFrame()
   # index = pd.MultiIndex.from_tuples([], names=["Time", "Height"])
   # dataframe = pd.DataFrame([[],[],[],[]], columns=[ "speedRadar","speedRadarDelta", "LidarValue"])
    def __init__(self, begin, end):
        self.dateBegin = begin
        self.dateEnd = end
        self.hours = np.arange(begin, end, timedelta(hours=0.5)).astype(datetime)
        self.days = np.arange(begin, end, timedelta(days=1)).astype(datetime)
        self.createDataframe()
    def createDataframe(self):
        for hour in self.hours:
            for height in self.heightGrid:
                entry = [(hour, height)]
                self.dataframe = self.dataframe.append(entry, ignore_index=True)
        self.dataframe = self.dataframe.rename(columns={0: "time", 1: "height", 2: "speedRadar", 3: "speedDeltaRadar", 4: "speedLidar", 5: "sppedDeltaLidar", 6: "speedDifference"})
        self.dataframe = self.dataframe.set_index(['time', 'height'])
    def mergeHeight(self,targetHeightList):
        heightGridDf = pd.DataFrame({'height': self.heightGrid})
        targetHeightDf = pd.DataFrame({'height': targetHeightList})
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(targetHeightDf)
        radarMatchIndex = knn.kneighbors(heightGridDf, return_distance=False)
        return radarMatchIndex
    def mergeTime(self,targetTimeList,gridTime):
        gridTimeSeries = []
        for entry in gridTime:
            gridTimeSeries.append(datetime.timestamp(entry))
        gridDf = pd.DataFrame({'time': gridTimeSeries})  
        #Target Grid
        targetTimestampList = []
        for entry in targetTimeList:
            stamp = (entry-2440587.5)*86400
            targetTimestampList.append(stamp)
        targetDf = pd.DataFrame({'time': targetTimestampList})
        #KNN
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(targetDf)
        matchList = knn.kneighbors(gridDf, return_distance=True)
        #generate result
        resultList = []
        for i in range(len(gridTimeSeries)):
            iMatch = int(matchList[1][i])
            if matchList[0][i] <= 60*14: #14 Minuten
                resultList.append(iMatch)
            else:
                resultList.append(np.nan)
        return resultList
    def readFile(self,path,radar, dailyHourList):
        if os.path.exists(path):
            dataset = Dataset(path, mode='r')
            height = dataset.variables['height'][:]
            speed = dataset.variables['speed'][:]
            speed = speed.filled(np.nan)
            speedDelta = dataset.variables['delta_speed'][:]
            speedDelta = speedDelta.filled(np.nan)
            time = dataset.variables['time'][:]
            if radar == True:
                speed = speed.T
                speedDelta = speedDelta.T
            matchHeightIndex = self.mergeHeight(height)
            matchTimeIndex = self.mergeTime(time,dailyHourList)
            for nHour in range(len(dailyHourList)):
                for i in range(len(self.heightGrid)):
                    iMatch = matchHeightIndex[i]
                    nMatch = matchTimeIndex[nHour]
                    if not math.isnan(iMatch) and not math.isnan(nMatch):
                        if radar == True:
                            self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'speedRadar'] = speed[iMatch,nMatch]
                            self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'speedDeltaRadar'] = speedDelta[iMatch,nMatch]
                        else:
                            self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'speedLidar'] = speed[iMatch,nMatch]
                            self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'speedDeltaLidar'] = speedDelta[iMatch,nMatch]
    def importDataset(self):
        for day in self.days:
            dailyHourList = []
            for hour in self.hours:
                if hour.strftime("%Y") == day.strftime("%Y"):
                    if hour.strftime("%m") == day.strftime("%m"):
                        if hour.strftime("%d") == day.strftime("%d"):
                            dailyHourList.append(hour)
            pathRadar = '/data/obs/site/jue/joyrad35/wind_ppi/data/'+day.strftime("%Y")+'/'+day.strftime("%m")+'/'+day.strftime("%Y%m%d")+'_joyrad35_wind_profile.nc'
            self.readFile(pathRadar, True, dailyHourList)
            pathLidar = '/data/obs/site/jue/wind_lidar/l1/'+day.strftime("%Y")+'/'+day.strftime("%m")+'/'+day.strftime("%d")+'/wind_vad-36_'+day.strftime("%Y%m%d")+'.nc'
            self.readFile(pathLidar, False, dailyHourList)            
    def calculateDifferences(self):
        for hour in self.hours:
            for height in self.heightGrid:
                radarValue = self.dataframe.loc[(hour,height),'speedRadar']
                lidarValue = self.dataframe.loc[(hour,height),'speedLidar']
                if not math.isnan(radarValue) and not math.isnan(lidarValue):
                    diff = radarValue - lidarValue
                    self.dataframe.loc[(hour,height),'speedDifference'] = diff
    def getHeightProfile(self):
        result = pd.DataFrame()
        for height in self.heightGrid:
            lidarSum = 0
            radarSum = 0
            bothSum = 0
            for hour in self.hours:
                radarValue = self.dataframe.loc[(hour,height),'speedRadar']
                lidarValue = self.dataframe.loc[(hour,height),'speedLidar']
                if not math.isnan(radarValue):
                    radarSum += 1
                if not math.isnan(lidarValue):
                    lidarSum += 1
                if not math.isnan(radarValue) and not math.isnan(lidarValue):
                    bothSum +=1
            radarCoverage = radarSum/len(self.hours)*100
            lidarCoverage = lidarSum/len(self.hours)*100
            #bothSum = bothSum/len(self.hours)*100
            totalCoverage = (radarSum+lidarSum-bothSum)/len(self.hours)*100
            entry = [(height, radarCoverage, lidarCoverage, totalCoverage)]
            result = result.append(entry, ignore_index=True)
        result = result.rename(columns={0: "height", 1: "radar Coverage", 2: "lidar Coverage", 3: "total Coverage"})
        result = result.set_index(['height'])
        return result




