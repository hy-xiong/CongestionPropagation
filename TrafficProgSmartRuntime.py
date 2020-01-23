
# coding: utf-8

# In[1]:

import os, time, datetime, glob, copy, sys, math
import numpy as np
import pandas as pd
from ult import *
from matplotlib import pyplot as plt

gct = time.clock
runtime = lambda et, st: str(datetime.timedelta(seconds=round(et-st)))

runNumber = int(sys.argv[1])
netSizeRatio = float(sys.argv[2])
p_thd = float(sys.argv[3])
delta_t = int(sys.argv[4])
tInt = int(sys.argv[5])
outDir = sys.argv[6]

countT = []
# In[2]:
noUTurn = 1
# read road network
allSelectedRids = np.genfromtxt('/Shared/xunzhou/Shenzhen_new/haoyi/CongProg/selectedRoadsCC%s' % ('_noUTurn' if noUTurn else ''), dtype=np.int32)

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
selectedRids = np.array(roads.keys(), dtype=np.int32)
connectedSets = []
for rid in roads:
    ncs = set(roads[rid].nextRids) | set(roads[rid].preRids)
    ncs.add(rid)
    mergeCSIndex = []
    newConnectedSets = []
    for j in xrange(len(connectedSets)):
        cs = connectedSets[j]
        if len(ncs & cs) > 0:
            mergeCSIndex.append(j)
        else:
            newConnectedSets.append(cs)
    for index in mergeCSIndex:
        ncs |= connectedSets[index]
    newConnectedSets.append(ncs)
    connectedSets = newConnectedSets
LCS = max(connectedSets, key=lambda x:len(x))
cRids = roads.keys()
for rid in cRids:
    if rid not in LCS:
        del roads[rid]
    else:
        roads[rid].preRids = list(set(roads[rid].preRids) & LCS)
        roads[rid].nextRids = list(set(roads[rid].nextRids) & LCS)
# vary network size
selectedRids = set([list(LCS)[0]])
maxSize = int(len(LCS) * netSizeRatio)
usedRids = set()
while(len(selectedRids)<maxSize):
    addedRids = selectedRids - usedRids
    for rid in addedRids:
        selectedRids |= set(roads[rid].preRids)
        selectedRids |= set(roads[rid].nextRids)
    usedRids.add(rid)
cRids = roads.keys()
for rid in cRids:
    if rid not in selectedRids:
        del roads[rid]
    else:
        roads[rid].preRids = list(set(roads[rid].preRids) & selectedRids)
        roads[rid].nextRids = list(set(roads[rid].nextRids) & selectedRids)
selectedRids = np.sort(np.array(list(selectedRids)))
netSize = selectedRids.shape[0]

st = gct()
# reconstruct road network adjacency to exclude those short-length linking roads during propagation
linkRoadMaxLength = 30 # meters
modAdjPrePath = {}
modAdjNextPath = {}
linkRoads = set()
for rid in roads:
    if roads[rid].length > linkRoadMaxLength:
        new_preRids = set()
        for nRid in roads[rid].preRids:
            if roads[nRid].length <= linkRoadMaxLength:
                newAdj = RoadNet.getPossibleNeighbors(rid, linkRoadMaxLength, 'pre', roads)
                for newAdj_rid in newAdj:
                    modAdjPrePath[(rid, newAdj_rid)] = newAdj[newAdj_rid]
                new_preRids |= set(newAdj.keys())
            else:
                new_preRids.add(nRid)
        roads[rid].preRids = list(new_preRids)

        new_nextRids = set()
        for nRid in roads[rid].nextRids:
            if roads[nRid].length <= linkRoadMaxLength:
                newAdj = RoadNet.getPossibleNeighbors(rid, linkRoadMaxLength, 'next', roads)
                for newAdj_rid in newAdj:
                    modAdjNextPath[(rid, newAdj_rid)] = newAdj[newAdj_rid]
                new_nextRids |= set(newAdj.keys())
            else:
                new_nextRids.add(nRid)
        roads[rid].nextRids = list(new_nextRids)
    else:
        linkRoads.add(rid)
countT.append(gct()-st)
# # Build congestion propagation index

# In[4]:

st = gct()
# parameters for building
nTrainDays = 23
tb_wd = [6, 10, 15, 20, 24]
def getTimeSlot(tDelta, tInt):
    return int(math.ceil(tDelta/pd.Timedelta(tInt, unit='m')))
tb_wd = [pd.Timedelta(v, 'h') for v in tb_wd]
tb_wd = [getTimeSlot(tb_wd[i] - tb_wd[0], tInt) for i in xrange(len(tb_wd))]
indices_congProg = ['weekday_%d-%d' % (tb_wd[i], tb_wd[i+1]-1) for i in xrange(len(tb_wd)-1)]
indices_congProg.extend(['Saturday', 'Sunday'])
print tb_wd, indices_congProg
countT.append(gct()-st)

# In[5]:

# read all congestion state of all road in all time slot in the entire dataset
ProcDataDir = '/Shared/xunzhou/Shenzhen_new/haoyi/CongProg/FinalSubmissionResult'
congestionStateRid = np.genfromtxt(os.path.join(ProcDataDir, 'congestionMatrixRoadIDByIndex'), dtype=np.uint32)
selectedRids_index = np.array([np.argwhere(congestionStateRid==rid).flatten()[0] for rid in selectedRids])
d = os.path.join(ProcDataDir, 'selRoads_congState_%d_%d' % (nTrainDays, tInt))
congStates = []
for f in sorted(glob.glob('%s/%s2014_11_*' % (d, 'noUTurn' if noUTurn else '')),
                key=lambda x: int(x.split('_')[-1]))[:nTrainDays]:
    congState = np.genfromtxt(f, delimiter=',')
    congStates.append(congState[selectedRids_index,:]) # allSelectedRids X tSlot

# In[6]:

st = gct()
# build index for each time period of weekday and each day of weekends given road connectivity
def updateCong(congStatM, congProgDict, indexRids, roads, linkRoads):
    jumpIndex = np.zeros(congStatM.shape, dtype=np.uint32)
    jumpIndex[:,:] = -1
    for ri in xrange(congStatM.shape[0]):
        mt = 0
        for t in xrange(congStatM.shape[1]):
            if congStatM[ri, t] == 1:
                for pt in xrange(mt, t+1):
                    jumpIndex[ri, pt] = t
                mt = t+1
    for ri in xrange(indexRids.shape[0]):
        rid = indexRids[ri]
        if rid not in linkRoads:
            for pRid in roads[rid].preRids:
                tInd = jumpIndex[ri,0]
                pRid_index = np.argwhere(indexRids == pRid).flatten()[0]
                while(tInd < congStatM.shape[1] - 1):
                    if tInd >= 0:
                        if congStatM[pRid_index,tInd]==0:
                            congProgDict[rid][pRid][1] += 1
                            if congStatM[pRid_index,tInd+1]==1:
                                congProgDict[rid][pRid][0] += 1
                        tInd = jumpIndex[ri,tInd+1]
                    else:
                        break

index_data_congProg = []
cDay = datetime.date(2014, 11, 1)
# set up index structure
for i in xrange(len(indices_congProg)):
    CongProg = {}
    for rid in selectedRids:
        if rid not in linkRoads:
            CongProg[rid] = {preRid:[0, 0] for preRid in roads[rid].preRids}
            CongProg[rid][rid] = [0, 0]
    index_data_congProg.append(CongProg)
# compute values for the index
for congState in congStates:
    if cDay.weekday() >= 5: # Saturday & Sunday
        CongProg = index_data_congProg[cDay.weekday()-1]
        congStatM = congState
        updateCong(congStatM, CongProg, selectedRids, roads, linkRoads)
    else: # weekday
        for j in xrange(len(tb_wd)-1):
            if j < len(tb_wd)-2:
                #add 1 more time slot for counting propagation
                congStatM = congState[:, tb_wd[j] : tb_wd[j+1] + 1]
            else:
                #end time of the day, cannot add more time slot to the end
                congStatM = congState[:, tb_wd[j] : tb_wd[j+1]]
            CongProg = index_data_congProg[j]
            updateCong(congStatM, CongProg, selectedRids, roads, linkRoads)
        congStatM = congState
    cDay += datetime.timedelta(days=1)
for i in xrange(len(indices_congProg)):
    CongProg = index_data_congProg[i]
    for rid in CongProg:
        for preRid in CongProg[rid]:
            if CongProg[rid][preRid][1] == 0:
                CongProg[rid][preRid] = 0.
            else:
                CongProg[rid][preRid] = CongProg[rid][preRid][0] * 1.0 / CongProg[rid][preRid][1]
countT.append(gct()-st)
# # compute traffic bottleneck score

st = gct()
# In[ ]:
def getCongIndexPos(day, tSlot, tb_wd):
    w = datetime.date(2014, 11, day).weekday()
    if w >= 5:
        return w - 1
    else:
        for tb_i in xrange(len(tb_wd[1:])):
            if tSlot < tb_wd[tb_i+1]:
                return tb_i

def checkProg(cRid, nRid, ct, CongStateM, rid_index):
    cRid_index = rid_index[cRid]
    nRid_index = rid_index[nRid]
    if (CongStateM[cRid_index][ct] == 1) and (CongStateM[nRid_index][ct] == 0) \
    and (CongStateM[nRid_index][ct+1] == 1):
        return 1
    else:
        return 0

def getConnCongSet(congRids, linkRoads, roads):
    CCSs = []
    for congRid in congRids:
        if congRid not in linkRoads:
            mergeIndex = set()
            for m in xrange(len(CCSs)):
                neighborhood = set(CCSs[m])
                for CCS_rid in CCSs[m]:
                    neighborhood |= set(roads[CCS_rid].preRids)
                if (len(set(roads[congRid].preRids) & CCSs[m]) > 0) or (congRid in neighborhood):
                    mergeIndex.add(m)
            newCCS = []
            mergeCCS = set([congRid])
            for m in xrange(len(CCSs)):
                if m in mergeIndex:
                    mergeCCS |= CCSs[m]
                else:
                    newCCS.append(CCSs[m])
            newCCS.append(mergeCCS)
            CCSs = newCCS
    return CCSs

# In[ ]:

# set parameter and read data
testDays = range(24, 31)
d = os.path.join(ProcDataDir, 'selRoads_congState_%d_%d' % (nTrainDays, tInt))
rid_index = {selectedRids[i]:i for i in xrange(selectedRids.shape[0])}
delta_t += 1
for testDay in testDays:
    f = '%s/%s2014_11_%d' % (d, 'noUTurn' if noUTurn else '', testDay)
    testCongState = np.genfromtxt(f, delimiter=',') # allSelectedRids X tSlot
    testCongState = testCongState[selectedRids_index, :]
    # make prediction for each day
    nTSlots = testCongState.shape[1]
    tMax = nTSlots - 1 if p_thd > 0 else nTSlots-delta_t-1
    for t in xrange(tMax):
        congRids = selectedRids[testCongState[:,t]==1]
        if p_thd > 0:
            t_thd = nTSlots-1
        else:
            if t + delta_t >= nTSlots - 1:
                t_thd = nTSlots - 1
            else:
                t_thd = t + delta_t
        # find connected congested sets, and roots and leaves
        CCSs = getConnCongSet(congRids, linkRoads, roads)
        # Naive solution
        roots_CCSs, paths_CCSs, visited_CCSs, rank_CCSs  = [], [], [], []
        for CCS in CCSs:
            roots, allPaths, allVisited = [], [], set()
            for congRid in CCS:
                # starting a BFS search from a given congested segment and compute the score for all segments
                isRoot = True
                for preRid in roads[congRid].preRids:
                    if preRid not in CCS:
                        isRoot = False
                progIndex = index_data_congProg[getCongIndexPos(testDay, t, tb_wd)]
                if isRoot:
                    roots.append(congRid)
                else:
                    paths, path_prob, ext_flags = [[congRid]], [1.0], [True]
                    maxTF = t+1
                    visitedRids = set([congRid])
                    while (np.any(ext_flags) and maxTF < t_thd):
                        progIndex = index_data_congProg[getCongIndexPos(testDay, maxTF-1, tb_wd)]
                        new_paths, new_path_prob, new_ext_flags = [], [], []
                        for m in xrange(len(paths)):
                            updated = False
                            if ext_flags[m]:
                                cl_rid = paths[m][-1]
                                p = path_prob[m]
                                for pre_cl_rid in roads[cl_rid].preRids:
                                    if (pre_cl_rid not in visitedRids) and (pre_cl_rid not in CCS):
                                        new_p = p * progIndex[cl_rid][pre_cl_rid]
                                        if (p_thd > 0 and new_p >= p_thd) or (p_thd == 0):
                                            subPath = copy.copy(paths[m])
                                            subPath.append(pre_cl_rid)
                                            new_paths.append(subPath)
                                            new_path_prob.append(new_p)
                                            new_ext_flags.append(True)
                                            updated = True
                            if not updated:
                                new_paths.append(paths[m])
                                new_path_prob.append(path_prob[m])
                                new_ext_flags.append(False)
                        paths, path_prob, ext_flags = new_paths, new_path_prob, new_ext_flags
                        visitedRids |= set([pa[-1] for pa in paths])
                        maxTF += 1
                    maxTF -= 1
                    allVisited |= visitedRids
                    allPaths.append(paths)
            roots_CCSs.append(roots)
            paths_CCSs.append(allPaths)
            visited_CCSs.append(allVisited)
countT.append(gct()-st)

if not os.path.exists(outDir):
    os.makedirs(outDir)
with open('%s/run%d' % (outDir, runNumber), 'w') as wrt:
    s = 'paramSetting: netSizeRatio=%.2f p_thd=%.5f tInt=%d\n' % (netSizeRatio, p_thd, tInt)
    s += '#_segments:%d\n' % netSize
    s += 'indexRuntime:%.4f\n' % countT[-2]
    s += 'totalRuntime:%.4f\n' % sum(countT)
    s += 'totalTimeSlots: %d' % ((nTSlots-1) * len(testDays))
    wrt.write(s)
