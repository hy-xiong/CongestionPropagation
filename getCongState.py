import pandas as pd
import numpy as np
import os, sys, time, datetime
from ult import RoadNet

noUTurn = int(sys.argv[1])
nMinVel = int(sys.argv[2])
pctNoise = int(sys.argv[3])
tInt = int(sys.argv[4])
obsDays = int(sys.argv[5])
saveRoadID = int(sys.argv[6])
f = sys.argv[7]

gct = time.time
runtime = lambda et, st: str(datetime.timedelta(seconds=round(et-st)))


if tInt % 5 != 0:
    raise Exception('time interval must be multiplication of 5. Input is %d' % tInt)

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
if saveRoadID:
    np.savetxt('/Shared/xunzhou/Shenzhen_new/haoyi/CongProg/FinalSubmissionResult/congestionMatrixRoadIDByIndex', selectedRids, fmt='%d')
print "Runtime %s" % runtime(gct(), st)

# read road spd in each day
df = pd.read_csv(f, dtype={'timeSlot':np.uint16, 'road':np.int32, 'avgGPS':np.float64, 
                           'nGPSPts':np.uint32, 'avgVehicles':np.float64, 'nVehicles':np.uint32})
df = df[df['road'].isin(selectedRids)]
if tInt > 5:
    df['newTSlot'] = df['timeSlot'].apply(lambda x:int(x*5/tInt))
    df['cumuVehSpd'] = df['avgVehicles'] * df['nVehicles']
    nTSlots = df['newTSlot'].max()+1
    gp = df.groupby(['road', 'newTSlot'])
    newDF = gp['cumuVehSpd', 'nVehicles'].sum()
    newDF['avgVehicles'] = newDF['cumuVehSpd'] / newDF['nVehicles'] 
else:
    nTSlots = df['timeSlot'].max()+1
    newDF = df
    newDF = newDF.set_index(['road', 'timeSlot'])

# tInt = int(os.path.basename(f).split('_')[2])
# stDayTime = pd.Timestamp(f[-29:-19].replace('_', '-'))
# stTime = stDayTime + pd.Timedelta(6, unit='h')

# read free flow spd
road_free_flow_speed = pd.read_csv('/Shared/xunzhou/Shenzhen_new/haoyi/CongProg/FinalSubmissionResult/obs_free_flow%s_%d_%d_%d' %\
                                   ('_noUTurn' if noUTurn else '', nMinVel, pctNoise, obsDays), 
                                   dtype={'roadID':np.int32, 'ff_spd':np.float64}, index_col='roadID')
road_free_flow_speed = road_free_flow_speed['ff_spd']

CongStat = []
for rid in selectedRids:
    CongStatRow = []
    if rid not in road_free_flow_speed:
        CongStatRow = [0 for t in xrange(nTSlots)]
    else:
        for t in xrange(nTSlots):
            if (rid, t) not in newDF.index:
                CongStatRow.append(0)
            else:
                ff_spd = road_free_flow_speed[rid]
                if 1 - newDF.loc[(rid,t)]['avgVehicles'] / ff_spd < 0.5:
                    CongStatRow.append(0)
                else:
                    CongStatRow.append(1)
    CongStat.append(CongStatRow)
CongStat = np.array(CongStat, dtype=np.uint8) # roadID X tSLotID
outDir = '/Shared/xunzhou/Shenzhen_new/haoyi/CongProg/FinalSubmissionResult/selRoads_congState_%d_%d' % (obsDays, tInt)
if not os.path.exists(outDir):
    os.mkdir(outDir)
np.savetxt(os.path.join(outDir, '%s%s' % ('noUTurn' if noUTurn else '', '_'.join(f.split('_')[-3:]))), 
           CongStat, fmt='%d', delimiter=',')