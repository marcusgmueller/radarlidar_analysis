# radarlidar_analysis
## Scientific Background
## Methods/Annahmen
## Code Base
This project consists of three files:
### RadarLidarWindSpeed.py
The file "RadarLidarWindSpeed.py" contains a class "RadarLidarWindSpeed". This class stores all data in a pandas dataframe. To access and change this data, a set of methods is provided:
- **createDataframe()** This methods creates a pandas dataframe to store all prcoessed data.
- **mergeHeight()** This method uses a neares neighbour algorith to map two height scales.
- **mergeTime()** This method uses a neares neighbour algorith to map two time scales.
- **readSpeedFile()** This method reads the file with radar and lidar wind speed data.
- **readDirectionFile()** This method reads the file with radar and lidar wind speed data.
- **importDataset()** This method summs up the import process and also calls readDirectionFile() and readDirectionFile().
- **calculateDifferences()** This method calculates the differences between radar and lidar data.
- **getSpeedHeightProfile()** This method calculates a coverage-height-profile for wind speed.
- **getDirectionHeightProfile()** This method calculates a coverage-height-profile for wind direction.
- **getCoverageHeightTimeSeries()** This method calculates a coverage-height-timeseries for data coverage.
- **calculateSpeedFusion()** This methods merges radar and lider speed data. 
- **calculateDirectionFusion()** This methods merges radar and lidar direction data.
- **calculateAvailability()** This methods calculates a value [0,1,2, 3] for data availability.
- **exportNCDF()** This method exports all data from the datafram into a netcdf file.
- **windspeedFullHeightCoveragePlot()** This method creates a speed coverage plot for the full height.
- **windspeedFullHeightOverviewPlot()**  This method creates a speed overview plot for the full height.
- **windspeedBoundaryLayerCoveragePlot()**  This method creates a speed coverage plot for the boundary layer.
- **windspeedBoundaryLayerOverviewPlot()** This method creates a speed overview plot for the boundary layer.
- **winddirectionFullHeightOverviewPlot()** This method creates a direction overview plot for the full height.
- **winddirectionBoundaryLayerOverviewPlot()** This method creates a direction overview plot for the boundary layer.
- **availabilityPlot()** This method creats an availability plot.
- **histogramDifferencePlot()** This method creates a histogram of the differences.

In addition to this class two files are provided to run routines automaticly:
### call_routine.py 
This code creates Quicklook plots for current and last day. It can be started by

`python3 call_routine.py`

The file can be modified in the following way. With the line

`days_range = (0, 2)`

the number of days, for which quicklooks are created, is selected. In this case, quicklooks are created for the current and the previous day. With the line 

`storagePath = "/work/marcus_mueller/routine/"`

the storage path can be changed. The software automaticly creates a folder structure based on the date under this path.
### call_historyPlot.py create statistical plots on data availability
