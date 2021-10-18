# radarlidar_analysis
## Scientific Background
## Methods/Annahmen
## Code Base
This project consists of three files:
### RadarLidarWindSpeed.py
The file "RadarLidarWindSpeed.py" contains a class "RadarLidarWindSpeed". This class stores all data in a pandas dataframe. To access and change this data, a set of methods is provided:
- createDataframe()
- mergeHeight()
- mergeTime()
- readSpeedFile()
- readDirectionFile()
- importDataset()
- calculateDifferences()
- getSpeedHeightProfile()
- getDirectionHeightProfile()
- getCoverageHeightTimeSeries()
- calculateSpeedFusion()
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
