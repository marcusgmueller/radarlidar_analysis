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
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

class RadarLidarWindSpeed:
    # Variables
    dateBegin = datetime.timestamp(datetime.now())
    dateEnd = datetime.timestamp(datetime.now())
    days = []
    hours = []
    heightGrid = list(range(0,13000,26))#range(0,15012,36)
    dataframe = pd.DataFrame()
    def __init__(self, begin, end):
        self.dateBegin = begin
        self.dateEnd = end
        self.hours = np.arange(begin, end, timedelta(hours=0.5)).astype(datetime)
        self.days = np.arange(begin, end, timedelta(days=1)).astype(datetime)
        self.hours = np.append(self.hours, end)
        self.days = np.append(self.days, self.dateEnd)
        #self.heightGrid = grid
        self.createDataframe()
    def createDataframe(self):
        for hour in self.hours:
            for height in self.heightGrid:
                entry = [(hour, height)]
                self.dataframe = self.dataframe.append(entry, ignore_index=True)
        self.dataframe = self.dataframe.rename(columns={0: "time", 1: "height", 2: "speedRadar", 3: "speedDeltaRadar", 4: "speedLidar", 5: "sppedDeltaLidar", 6: "speedDifference", 7: "speedFusion", 8: "directionRadar", 9: "directionDeltaRadar", 10: "directionLidar", 11: "directionDeltaLidar", 12: "directionDifference", 13: "directionFusion", 14: "availability", 15: "missingRadarFile", 16: "missingLidarFile", })
        self.dataframe["speedRadar"] = np.nan
        self.dataframe["speedLidar"] = np.nan
        self.dataframe["speedDeltaRadar"] = np.nan
        self.dataframe["sppedDeltaLidar"] = np.nan
        self.dataframe["speedDifference"] = np.nan
        self.dataframe["speedFusion"] = np.nan
        self.dataframe["directionRadar"] = np.nan
        self.dataframe["directionLidar"] = np.nan
        self.dataframe["directionDeltaRadar"] = np.nan
        self.dataframe["directionDeltaLidar"] = np.nan
        self.dataframe["directionDifference"] = np.nan
        self.dataframe["directionFusion"] = np.nan
        self.dataframe["availability"] = np.nan
        self.dataframe["missingRadarFile"] = np.nan
        self.dataframe["missingLidarFile"] = np.nan
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
            stamp = (entry-2440587.5)*86400-60*60
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
    def readSpeedFile(self,path,radar, dailyHourList):
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
        else:
            for nHour in range(len(dailyHourList)):
                for i in range(len(self.heightGrid)):
                    if radar == True:
                        self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'missingRadarFile'] = True
                    else:
                        self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'missingLidarFile'] = True
    def readDirectionFile(self,path,radar, dailyHourList):
        if os.path.exists(path):
            dataset = Dataset(path, mode='r')
            height = dataset.variables['height'][:]
            speed = dataset.variables['dir'][:]
            speed = speed.filled(np.nan)
            speedDelta = dataset.variables['delta_dir'][:]
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
                            self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'directionRadar'] = speed[iMatch,nMatch]
                            self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'directionDeltaRadar'] = speedDelta[iMatch,nMatch]
                        else:
                            self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'directionLidar'] = speed[iMatch,nMatch]
                            self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'directionDeltaLidar'] = speedDelta[iMatch,nMatch]
        else:
            for nHour in range(len(dailyHourList)):
                for i in range(len(self.heightGrid)):
                    if radar == True:
                        self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'missingRadarFile'] = True
                    else:
                        self.dataframe.loc[(dailyHourList[nHour],self.heightGrid[i]),'missingLidarFile'] = True
    def importDataset(self):
        for day in self.days:
            dailyHourList = []
            for hour in self.hours:
                if hour.strftime("%Y") == day.strftime("%Y"):
                    if hour.strftime("%m") == day.strftime("%m"):
                        if hour.strftime("%d") == day.strftime("%d"):
                            dailyHourList.append(hour)
            pathRadar = '/data/obs/site/jue/joyrad35/wind_ppi/data/'+day.strftime("%Y")+'/'+day.strftime("%m")+'/'+day.strftime("%Y%m%d")+'_joyrad35_wind_profile.nc'
            self.readSpeedFile(pathRadar, True, dailyHourList)
            self.readDirectionFile(pathRadar, True, dailyHourList)
            pathLidar = '/data/obs/site/jue/wind_lidar/l1/'+day.strftime("%Y")+'/'+day.strftime("%m")+'/'+day.strftime("%d")+'/wind_vad-36_'+day.strftime("%Y%m%d")+'.nc'
            self.readSpeedFile(pathLidar, False, dailyHourList)
            self.readDirectionFile(pathLidar, False, dailyHourList)            
    def calculateDifferences(self):
        for hour in self.hours:
            for height in self.heightGrid:
                radarSpeedValue = self.dataframe.loc[(hour,height),'speedRadar']
                lidarSpeedValue = self.dataframe.loc[(hour,height),'speedLidar']
                radarDirectionValue = self.dataframe.loc[(hour,height),'directionRadar']
                lidarDirectionValue = self.dataframe.loc[(hour,height),'directionLidar']
                if not math.isnan(radarSpeedValue) and not math.isnan(lidarSpeedValue):
                    diffSpeed = radarSpeedValue - lidarSpeedValue
                    self.dataframe.loc[(hour,height),'speedDifference'] = diffSpeed
                    diffDir = radarDirectionValue - lidarDirectionValue
                    self.dataframe.loc[(hour,height),'directionDifference'] = diffDir
    def getSpeedHeightProfile(self):
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
    def getDirectionHeightProfile(self):
        result = pd.DataFrame()
        for height in self.heightGrid:
            lidarSum = 0
            radarSum = 0
            bothSum = 0
            for hour in self.hours:
                radarValue = self.dataframe.loc[(hour,height),'directionRadar']
                lidarValue = self.dataframe.loc[(hour,height),'directionLidar']
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
    def getCoverageHeightTimeSeries(self):
        result = pd.DataFrame()
        for day in self.days:
            allHeights = 0
            for height in self.heightGrid:
                lidarSum = 0
                radarSum = 0
                bothSum = 0
                counter = 0
                for hour in self.hours:
                    if hour.strftime("%Y") == day.strftime("%Y"):
                        if hour.strftime("%m") == day.strftime("%m"):
                            if hour.strftime("%d") == day.strftime("%d"):
                                counter += 1
                                radarValue = self.dataframe.loc[(hour,height),'speedRadar']
                                lidarValue = self.dataframe.loc[(hour,height),'speedLidar']
                                missingRadarFile = self.dataframe.loc[(hour,height),'missingRadarFile']
                                missingLidarFile = self.dataframe.loc[(hour,height),'missingLidarFile']
                                if not math.isnan(radarValue):
                                    radarSum += 1
                                if not math.isnan(lidarValue):
                                    lidarSum += 1
                                if not math.isnan(radarValue) and not math.isnan(lidarValue):
                                    bothSum +=1
                radarCoverage = radarSum/counter*100
                lidarCoverage = lidarSum/counter*100
                if missingLidarFile == True:
                    lidarCoverage = np.nan
                if missingRadarFile == True:
                    radarCoverage = np.nan
                totalCoverage = (radarSum+lidarSum-bothSum)/counter*100
                entry = [(height, day, radarCoverage, lidarCoverage, totalCoverage)]
                result = result.append(entry, ignore_index=True)
        result = result.rename(columns={0: "height",1: "day", 2: "radar Coverage", 3: "lidar Coverage", 4: "total Coverage"})
        return result  
    def calculateSpeedFusion(self):
        for hour in self.hours:
            for height in self.heightGrid:
                radarValue = self.dataframe.loc[(hour,height),'speedRadar']
                lidarValue = self.dataframe.loc[(hour,height),'speedLidar']
                radarDelta = self.dataframe.loc[(hour,height),'speedDeltaRadar']
                lidarDelta = self.dataframe.loc[(hour,height),'speedDeltaLidar']
                if not math.isnan(radarValue) and not math.isnan(lidarValue) and not math.isnan(radarDelta) and not math.isnan(lidarDelta):
                    fusion = (radarValue*radarDelta+lidarValue*lidarDelta)/(radarDelta+lidarDelta)
                    self.dataframe.loc[(hour,height),'speedFusion'] = fusion
                elif math.isnan(radarValue) and not math.isnan(lidarValue):
                    self.dataframe.loc[(hour,height),'speedFusion'] = lidarValue
                elif math.isnan(lidarValue) and not math.isnan(radarValue):
                    self.dataframe.loc[(hour,height),'speedFusion'] = radarValue
    def calculateDirectionFusion(self):
        for hour in self.hours:
            for height in self.heightGrid:
                radarValue = self.dataframe.loc[(hour,height),'directionRadar']
                lidarValue = self.dataframe.loc[(hour,height),'directionLidar']
                radarDelta = self.dataframe.loc[(hour,height),'directionDeltaRadar']
                lidarDelta = self.dataframe.loc[(hour,height),'directionDeltaLidar']
                if not math.isnan(radarValue) and not math.isnan(lidarValue) and not math.isnan(radarDelta) and not math.isnan(lidarDelta):
                    fusion = (radarValue*radarDelta+lidarValue*lidarDelta)/(radarDelta+lidarDelta)
                    self.dataframe.loc[(hour,height),'directionFusion'] = fusion
                elif math.isnan(radarValue) and not math.isnan(lidarValue):
                    self.dataframe.loc[(hour,height),'directionFusion'] = lidarValue
                elif math.isnan(lidarValue) and not math.isnan(radarValue):
                    self.dataframe.loc[(hour,height),'directionFusion'] = radarValue
    def calculateAvailability(self):
        # no data = 0
        # only radar = 1
        # only lidar = 2
        # both = 3
        for hour in self.hours:
            for height in self.heightGrid:
                radarValue = self.dataframe.loc[(hour,height),'speedRadar']
                lidarValue = self.dataframe.loc[(hour,height),'speedLidar']
                if not math.isnan(radarValue) and not math.isnan(lidarValue):
                    self.dataframe.loc[(hour,height),'availability'] = 3
                elif math.isnan(radarValue) and not math.isnan(lidarValue):
                    self.dataframe.loc[(hour,height),'availability'] = 2
                elif math.isnan(lidarValue) and not math.isnan(radarValue):
                    self.dataframe.loc[(hour,height),'availability'] = 1
                else:
                    self.dataframe.loc[(hour,height),'availability'] = 0
    def exportNCDF(self, plotFilePath):
        x_array = self.dataframe.to_xarray()
        path = plotFilePath+str(self.dateBegin.strftime("%Y/%m/%d/"))+'netcdf'+str(self.dateBegin.strftime("%Y%m%d"))+".nc"
        x_array.to_netcdf(path=path)
    def windspeedFullHeightCoveragePlot(self, plotFilePath):
        result = self.getSpeedHeightProfile()
        plt.figure(figsize=(20,10))
        ax1 = plt.axes()
        plt.plot(result['radar Coverage'].tolist(),self.heightGrid, 'go-', label='Radar')
        plt.plot(result['lidar Coverage'].tolist(),self.heightGrid, 'rs-', label='Lidar')
        plt.plot(result['total Coverage'].tolist(),self.heightGrid, 'b*-', label='Total')
        ax1.set_ylim([0, 12000])
        ax1.set_xlim([0, 100])
        plt.xlabel("coverage [%]", fontsize=16)
        plt.ylabel("height [m]", fontsize=16)
        plt.legend(fontsize=16)
        plt.title('data coverage by height: '+self.dateBegin.strftime("%Y-%m-%d"), fontsize=16)
        plt.savefig(plotFilePath+str(self.dateBegin.strftime("%Y/%m/%d/"))+"coverageHeightPlot_fullheight_"+str(self.dateBegin.strftime("%Y%m%d"))+".png",dpi=150)
    def windspeedFullHeightOverviewPlot(self, plotFilePath):    
        df = self.dataframe.copy(deep=True)
        df.reset_index(level=0, inplace=True)
        df.reset_index(level=0, inplace=True)
        diff = df.pivot(index="height", columns="time", values="speedDifference")
        fusion = df.pivot(index="height", columns="time", values="speedFusion")
        availability = df.pivot(index="height", columns="time", values="availability")
        X,Y = np.meshgrid(self.hours, self.heightGrid)
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True, sharey=False)

        fig.suptitle("data overview "+self.dateBegin.strftime("%Y-%m-%d"), fontsize=16)

        # Availability
        col_dict={1:"blue", 2:"red", 13:"orange", 7:"green"}
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        im2 = axes[0].pcolor(X,Y,availability,cmap=cm, vmin=0, vmax=3)
        axes[0].set_title("data availability ")
        axes[0].set_ylabel("height AGL [m]")

        # Difference
        im3 = axes[1].pcolor(X,Y,diff,cmap='bwr', vmin=-10, vmax=10)#+-10
        axes[1].set_title("Difference: Radar - Lidar ")
        axes[1].set_ylabel("height AGL [m]")

        # Fusion
        array = fusion.to_numpy()
        maxValue = np.nanmax(array)
        if maxValue < 30.0:
            maxValue = 30.0
        im = axes[2].pcolor(X,Y,fusion,cmap='hsv', vmin=0, vmax=maxValue)
        axes[2].set_title("Radar/Lidar fusion ")
        axes[2].set_xlabel("Time UTC [h]")
        axes[2].set_ylabel("height AGL [m]")
        #plt.savefig(plotFilePath+"merge_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300)
        # cbar speed
        #cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        #cbar = fig.colorbar(im, cax=cb_ax)
        #cbar.set_label('Horizontal wind speed [m/s]')
        cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.set_label('Horizontal wind speed [m/s]')



        # cbar difference
        cb_ax3 = fig.add_axes([1.1, 0.1, 0.02, 0.8])
        cbar3 = fig.colorbar(im3, cax=cb_ax3)
        cbar3.set_label('Difference [m/s]')

        # cbar availability
        cb_ax2 = fig.add_axes([1.2, 0.1, 0.02, 0.8])
        cbar2 = fig.colorbar(im2, cax=cb_ax2, ticks=[0,1,2,3])
        cbar2.set_ticks([0,1,2,3])
        cbar2.set_ticklabels(["no data", "only Radar", "only Lidar","both"])
        cbar2.set_label('data availability')

        xformatter = mdates.DateFormatter('%H:%M')
        plt.gcf().axes[2].xaxis.set_major_formatter(xformatter)

        #fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.4)
        plt.savefig(plotFilePath+str(self.dateBegin.strftime("%Y/%m/%d/"))+"merge_windspeed_fullheight_"+str(self.dateBegin.strftime("%Y%m%d"))+".png",dpi=150,bbox_inches='tight')
    def windspeedBoundaryLayerCoveragePlot(self, plotFilePath):
        result = self.getSpeedHeightProfile()
        plt.figure(figsize=(20,10))
        fig1 = plt.figure(figsize=(20,10))
        ax1 = plt.axes()
        ax1.plot(result['radar Coverage'].tolist(),self.heightGrid, 'go-', label='Radar')
        ax1.plot(result['lidar Coverage'].tolist(),self.heightGrid, 'rs-', label='Lidar')
        ax1.plot(result['total Coverage'].tolist(),self.heightGrid, 'b*-', label='Total')
        ax1.set_ylim([0, 3000])
        ax1.set_xlim([0, 100])
        plt.xlabel("coverage [%]", fontsize=16)
        plt.ylabel("height [m]", fontsize=16)
        plt.legend(fontsize=16)
        plt.title('data coverage by height: '+self.dateBegin.strftime("%Y-%m-%d"), fontsize=16)
        plt.savefig(plotFilePath+str(self.dateBegin.strftime("%Y/%m/%d/"))+"coverageHeightPlot_boundarylayer_"+str(self.dateBegin.strftime("%Y%m%d"))+".png",dpi=150,bbox_inches='tight')
    def windspeedBoundaryLayerOverviewPlot(self, plotFilePath):
        #plot data overview
        df = self.dataframe.copy(deep=True)
        df.reset_index(level=0, inplace=True)
        df.reset_index(level=0, inplace=True)
        diff = df.pivot(index="height", columns="time", values="speedDifference")
        fusion = df.pivot(index="height", columns="time", values="speedFusion")
        availability = df.pivot(index="height", columns="time", values="availability")
        X,Y = np.meshgrid(self.hours, self.heightGrid)



        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True, sharey=False)

        fig.suptitle("data overview "+self.dateBegin.strftime("%Y-%m-%d"), fontsize=16)

        # Availability
        col_dict={1:"blue", 2:"red", 13:"orange", 7:"green"}
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        im2 = axes[0].pcolor(X,Y,availability,cmap=cm, vmin=0, vmax=3)
        axes[0].set_title("data availability ")
        axes[0].set_ylabel("height AGL [m]")
        axes[0].set_ylim([0, 3000])

        # Difference
        im3 = axes[1].pcolor(X,Y,diff,cmap='bwr', vmin=-10, vmax=10)#+-10
        axes[1].set_title("Difference: Radar - Lidar ")
        axes[1].set_ylabel("height AGL [m]")
        axes[1].set_ylim([0, 3000])

        # Fusion
        im = axes[2].pcolor(X,Y,fusion,cmap='hsv', vmin=0, vmax=30)
        axes[2].set_title("Radar/Lidar fusion ")
        axes[2].set_xlabel("Time UTC [h]")
        axes[2].set_ylabel("height AGL [m]")
        axes[2].set_ylim([0, 3000])
        #plt.savefig(plotFilePath+"merge_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300)
        # cbar speed
        #cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        #cbar = fig.colorbar(im, cax=cb_ax)
        #cbar.set_label('Horizontal wind speed [m/s]')
        cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.set_label('Horizontal wind speed [m/s]')



        # cbar difference
        cb_ax3 = fig.add_axes([1.1, 0.1, 0.02, 0.8])
        cbar3 = fig.colorbar(im3, cax=cb_ax3)
        cbar3.set_label('Difference [m/s]')

        # cbar availability
        cb_ax2 = fig.add_axes([1.2, 0.1, 0.02, 0.8])
        cbar2 = fig.colorbar(im2, cax=cb_ax2, ticks=[0,1,2,3])
        cbar2.set_ticks([0,1,2,3])
        cbar2.set_ticklabels(["no data", "only Radar", "only Lidar","both"])
        cbar2.set_label('data availability')

        xformatter = mdates.DateFormatter('%H:%M')
        plt.gcf().axes[2].xaxis.set_major_formatter(xformatter)

        #fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.4)
        plt.savefig(plotFilePath+str(self.dateBegin.strftime("%Y/%m/%d/"))+"merge_windspeed_boundarylayer_"+str(self.dateBegin.strftime("%Y%m%d"))+".png",dpi=150,bbox_inches='tight')
        plt.close()
        #plt.show()
    def winddirectionFullHeightOverviewPlot(self, plotFilePath):
        df = self.dataframe.copy(deep=True)
        df.reset_index(level=0, inplace=True)
        df.reset_index(level=0, inplace=True)
        diff = df.pivot(index="height", columns="time", values="directionDifference")
        fusion = df.pivot(index="height", columns="time", values="directionFusion")
        availability = df.pivot(index="height", columns="time", values="availability")
        X,Y = np.meshgrid(self.hours, self.heightGrid)



        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True, sharey=False)

        fig.suptitle("data overview "+self.dateBegin.strftime("%Y-%m-%d"), fontsize=16)

        # Availability
        col_dict={1:"blue", 2:"red", 13:"orange", 7:"green"}
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        im2 = axes[0].pcolor(X,Y,availability,cmap=cm, vmin=0, vmax=3)
        axes[0].set_title("data availability ")
        axes[0].set_ylabel("height AGL [m]")

        # Difference
        im3 = axes[1].pcolor(X,Y,diff,cmap='bwr', vmin=-20, vmax=20)#+-10
        axes[1].set_title("Difference: Radar - Lidar ")
        axes[1].set_ylabel("height AGL [m]")

        # Fusion
        im = axes[2].pcolor(X,Y,fusion,cmap='twilight', vmin=0, vmax=360)#vmax=32, cmap=viridis
        axes[2].set_title("Radar/Lidar fusion ")
        axes[2].set_xlabel("Time UTC [h]")
        axes[2].set_ylabel("height AGL [m]")
        #plt.savefig(plotFilePath+"merge_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300)
        # cbar speed
        #cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        #cbar = fig.colorbar(im, cax=cb_ax)
        #cbar.set_label('Horizontal wind speed [m/s]')
        cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.set_label('Horizontal wind direction [°]')



        # cbar difference
        cb_ax3 = fig.add_axes([1.1, 0.1, 0.02, 0.8])
        cbar3 = fig.colorbar(im3, cax=cb_ax3)
        cbar3.set_label('Difference [°]')#Difference [m/s]

        # cbar availability
        cb_ax2 = fig.add_axes([1.2, 0.1, 0.02, 0.8])
        cbar2 = fig.colorbar(im2, cax=cb_ax2, ticks=[0,1,2,3])
        cbar2.set_ticks([0,1,2,3])
        cbar2.set_ticklabels(["no data", "only Radar", "only Lidar","both"])
        cbar2.set_label('data availability')

        xformatter = mdates.DateFormatter('%H:%M')
        plt.gcf().axes[2].xaxis.set_major_formatter(xformatter)

        plt.savefig(plotFilePath+str(self.dateBegin.strftime("%Y/%m/%d/"))+"merge_winddirection_fullheight_"+str(self.dateBegin.strftime("%Y%m%d"))+".png",dpi=150,bbox_inches='tight')
    def winddirectionBoundaryLayerOverviewPlot(self, plotFilePath):
        df = self.dataframe.copy(deep=True)
        df.reset_index(level=0, inplace=True)
        df.reset_index(level=0, inplace=True)
        diff = df.pivot(index="height", columns="time", values="directionDifference")
        fusion = df.pivot(index="height", columns="time", values="directionFusion")
        availability = df.pivot(index="height", columns="time", values="availability")
        X,Y = np.meshgrid(self.hours, self.heightGrid)



        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True, sharey=False)

        fig.suptitle("data overview "+self.dateBegin.strftime("%Y-%m-%d"), fontsize=16)

        # Availability
        col_dict={1:"blue", 2:"red", 13:"orange", 7:"green"}
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        im2 = axes[0].pcolor(X,Y,availability,cmap=cm, vmin=0, vmax=3)
        axes[0].set_title("data availability ")
        axes[0].set_ylabel("height AGL [m]")
        axes[0].set_ylim([0, 3000])

        # Difference
        im3 = axes[1].pcolor(X,Y,diff,cmap='bwr', vmin=-20, vmax=20)#+-10
        axes[1].set_title("Difference: Radar - Lidar ")
        axes[1].set_ylabel("height AGL [m]")
        axes[1].set_ylim([0, 3000])

        # Fusion
        im = axes[2].pcolor(X,Y,fusion,cmap='twilight', vmin=0, vmax=360)#vmax=32, cmap=viridis
        axes[2].set_title("Radar/Lidar fusion ")
        axes[2].set_xlabel("Time UTC [h]")
        axes[2].set_ylabel("height AGL [m]")
        axes[2].set_ylim([0, 3000])
        #plt.savefig(plotFilePath+"merge_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300)
        # cbar speed
        #cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        #cbar = fig.colorbar(im, cax=cb_ax)
        #cbar.set_label('Horizontal wind speed [m/s]')
        cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.set_label('Horizontal wind direction [°]')



        # cbar difference
        cb_ax3 = fig.add_axes([1.1, 0.1, 0.02, 0.8])
        cbar3 = fig.colorbar(im3, cax=cb_ax3)
        cbar3.set_label('Difference [°]')#Difference [m/s]

        # cbar availability
        cb_ax2 = fig.add_axes([1.2, 0.1, 0.02, 0.8])
        cbar2 = fig.colorbar(im2, cax=cb_ax2, ticks=[0,1,2,3])
        cbar2.set_ticks([0,1,2,3])
        cbar2.set_ticklabels(["no data", "only Radar", "only Lidar","both"])
        cbar2.set_label('data availability')

        xformatter = mdates.DateFormatter('%H:%M')
        plt.gcf().axes[2].xaxis.set_major_formatter(xformatter)

        #fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.4)
        plt.savefig(plotFilePath+str(self.dateBegin.strftime("%Y/%m/%d/"))+"merge_winddirection_boundarylayer_"+str(self.dateBegin.strftime("%Y%m%d"))+".png",dpi=150,bbox_inches='tight')




