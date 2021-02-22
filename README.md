# radarlidar_analysis
## usage
### import class
```
from RadarLidarWindSpeed import RadarLidarWindSpeed
```
### create object
```
dateBegin = datetime(2020, 1, 1)
dateEnd = datetime(2020, 1, 31)
analysis = RadarLidarWindSpeed(dateBegin, dateEnd)
```
### import Dataset
```
analysis.importDataset()
```
### height profile
```
result = analysis.getHeightProfile()
```
### calculate differences
```
analysis.calculateDifferences()
```
### Coverage Height Time Series
```
analysis.getCoverageHeightTimeSeries()
```
### Radar Lidar Fusion
```
analysis.calculateFusion()
```
### Data Availability
```
analysis.calculateAvailability()
```
