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
### get height profile
```
result = analysis.getHeightProfile()
