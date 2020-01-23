import os, time, datetime, sys
import numpy as np
import pandas as pd
from ult import RoadNet

# parameters for finding free flow
nMinVehicles = int(sys.argv[1])
noisePercentile = int(sys.argv[2])
nTrainDays = int(sys.argv[3])
dstFolder = sys.argv[4]

gct = time.time
runtime = lambda et, st: str(datetime.timedelta(seconds=round(et-st)))

# read road network
# all road types
# typeOfRoads=["motorway","motorway_link","primary",
#              "primary_link","secondary","secondary_link",
#              "residential","service","trunk",
#              "trunk_link","tertiary","tertiary_link"]
noUTurn = 1
st = gct()
# read road network
allSelectedRids = np.genfromtxt('/Shared/xunzhou/Shenzhen_new/haoyi/CongProg/selectedRoadsCC%s' %\
                                ('_noUTurn' if noUTurn else ''), dtype=np.int32)

gridLen = 500.0 #500 meter grid
f_road ='/Shared/xunzhou/Shenzhen_new/haoyi/RoadNetwork/road_projected_utm50.geojson'
# select road types for analysis

RoadSelection = True
typeOfRoads=["motorway","motorway_link","primary",
             "primary_link", "trunk", "trunk_link"]

roads, topLeftOrigin, xyNGrids, grid_to_road, road_to_grid = RoadNet.readGeojson(f_road, gridLen,
                                                                                 selectedRoadTypes=typeOfRoads,
                                                                                 selectedRoadIDs=allSelectedRids.tolist(),
                                                                                 noUTurn = True)
# find largest connected component
selectedRids, linkRoads = RoadNet.find_LCC_update_Adj(roads, linkRoadMaxLength = 30)
print "Runtime %s" % runtime(gct(), st)


# read all GPS point speed
road_GPS_spd = pd.read_csv('/Shared/xunzhou/Shenzhen_new/haoyi/CongProg/RoadSpd/free_flow/road_GPS_spd_sort',
                           dtype={'device':np.uint32,'spd':np.uint16,'road':np.int32},
                           sep=',',
                           converters={'time': lambda x: pd.Timestamp(x)},
                           error_bad_lines=False)
road_GPS_spd = road_GPS_spd[road_GPS_spd['road'].isin(selectedRids)]
road_GPS_spd['day'] = road_GPS_spd['time'].apply(lambda x: x.day)
road_GPS_spd = road_GPS_spd[road_GPS_spd['day'] <= nTrainDays]
# find free flow
st = gct()
seg_st_index = road_GPS_spd['road'].diff()
seg_st_index = seg_st_index[seg_st_index != 0].index.values
if seg_st_index[-1] < road_GPS_spd.index.values[-1]:
    seg_st_index = np.append(seg_st_index, road_GPS_spd.index.values[-1]+1)
print 'runtime: find road index starting row index %s' % runtime(gct(), st)
st = gct()
road_free_flow_speed = {}
for i in xrange(seg_st_index.shape[0] - 1):
    road_spd_pts = road_GPS_spd.loc[seg_st_index[i] : seg_st_index[i+1] - 1]
    road_spd_pts = road_spd_pts[road_spd_pts['spd'] > 0.]
    nVeh = road_spd_pts['device'].unique().shape[0]
    if nVeh >= nMinVehicles:
        spds = road_spd_pts['spd'].values
#         print spds, spds.shape[0], road_GPS_spd['road'].loc[seg_st_index[i]], nVeh
        free_flow_spd = np.max(spds[spds < np.percentile(spds, 100 - noisePercentile)])
        roadID = road_GPS_spd['road'].loc[seg_st_index[i]]
        road_free_flow_speed[roadID] = free_flow_spd
print 'runtime: get free flow speed for roads satisfying conditions %s' % runtime(gct(), st)
print '# of roads having free flow: %d (%.3f%%)' % (len(road_free_flow_speed), 
                                                    len(road_free_flow_speed) * 100.0/selectedRids.shape[0])
road_free_flow_speed = pd.Series(road_free_flow_speed)
df_free_flow_spd = road_free_flow_speed.to_frame()
df_free_flow_spd.columns = ['ff_spd']
df_free_flow_spd.index.name = 'roadID'
df_free_flow_spd.to_csv(os.path.join('obs_free_flow_%d_%d' % (nMinVehicles, noisePercentile)))