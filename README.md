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
- **calculateSpeedFusion()** This methods merges radar and lider data. 
- calculateDirectionFusion()
- calculateAvailability()
- exportNCDF()
- windspeedFullHeightCoveragePlot()
- windspeedFullHeightOverviewPlot()
- windspeedBoundaryLayerCoveragePlot()
- windspeedBoundaryLayerOverviewPlot()
- winddirectionFullHeightOverviewPlot()
- winddirectionBoundaryLayerOverviewPlot()
- availabilityPlot()
- histogramDifferencePlot()

Class for processing Radar/Lidar data
### call_routine.py create Quicklook plots for current and last day
### call_historyPlot.py create statistical plots on data availability
