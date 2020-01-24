# CongestionPropagation
Key input files:
1. Road network: Geojson format, LineString feature collections, properties are:
- id: int
- preAdj: upstream adjacent road ID, String, comma delimited, e.g. "2, 18"
- nextAdj: downstream adjacent road ID, String
- roadType: OSM road type, e.g. motorway, String

2. Vehicle GPS data: .csv like table with comma delimited:

```text
road,time,device,spd
0,2014-11-01 06:03:26,1440575,41
0,2014-11-01 06:03:28,1440528,36
```

Scripts:
- getFreeFlow: get road segment free flow speed from historical data
- getCongState: get road segment binary congestion state given computed road speed data (derived from GPS)
- TrafficProgSmartRuntime.py: predict congestion propagation and evaluate running time

Jupyter Notebook Scripts:
There is many duplications between the jupyter scripts and raw python scripts. The jupyter notebook is for pattern exploreration. Once the code is fixed, .py file is created for distributed computing.
- ROC.ipynb uses the output from predict congestion propagation.ipynb to build ROC and AUC curve

If you would like to use this work, please cite:

```text
@inproceedings{xiong2018predicting,
  title={Predicting traffic congestion propagation patterns: a propagation graph approach},
  author={Xiong, Haoyi and Vahedian, Amin and Zhou, Xun and Li, Yanhua and Luo, Jun},
  booktitle={Proceedings of the 11th ACM SIGSPATIAL International Workshop on Computational Transportation Science},
  pages={60--69},
  year={2018}
}
```