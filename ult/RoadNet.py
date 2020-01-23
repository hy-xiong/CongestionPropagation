import math, json, pyproj, copy
from scipy.spatial import distance
import numpy as np

inBetween = lambda v, bot, top: True if v >= bot-0.001 and v <= top+0.001 else False
minMax = lambda v1, v2: (min(v1, v2), max(v1, v2))

prj1 = pyproj.Proj('+proj=latlong +datum=WGS84')
prj2 = pyproj.Proj('+proj=utm +zone=50 +datum=WGS84')
prj = lambda x,y: pyproj.transform(prj1, prj2, x, y)

class Road:
    def __init__(self, rid, nextRids, preRids, shape_pts, roadType):
        self.rid = rid
        self.nextRids = nextRids
        self.preRids = preRids
        self.shape_pts = shape_pts
        self.roadType = roadType
        self.length = 0.0
        for k in xrange(len(shape_pts) - 1):
            self.length += EuDist(shape_pts[k], shape_pts[k+1])
        self.visual_shape_pts = copy.deepcopy(shape_pts)
        xmin, ymin = self.visual_shape_pts[0]
        xmax, ymax = self.visual_shape_pts[0]
        for line_seg in self.visual_shape_pts:
            xmin = min(xmin, line_seg[0])
            ymin = min(ymin, line_seg[1])
            xmax = max(xmax, line_seg[0])
            ymax = max(ymax, line_seg[1])
        self.bbox = [xmin, ymin, xmax, ymax]
            
def EuDist(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return math.sqrt((x2 - x1)** 2 + (y2 - y1) ** 2)

def getGridNumForPoint(pt, xmin, ymax, gridLen):
    # left to right, top to bot
    nX = int((pt[0] - xmin) / gridLen)
    nY = int((ymax - pt[1]) / gridLen)
    return nX, nY

def _addToListDict(d, v, e):
    if v not in d:
        d[v] = [e]
    else:
        if e not in d[v]:
            d[v].append(e)

def _isLineOverlapSquare(line, SquareBBox):
    # @arg line: [[x, y], [x, y]]
    # @arg SquareBBox: [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = SquareBBox
    bBot = [[xmin, ymin], [xmax, ymin]]
    bLeft = [[xmin, ymin], [xmin, ymax]]
    bRight = [[xmax, ymin], [xmax, ymax]]
    bTop = [[xmin, ymax], [xmax, ymax]]
    def isIntersect(lineA, lineB):
        [x1, y1], [x2, y2] = lineA
        [x3, y3], [x4, y4] = lineB
        slopeA = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else "NA"
        slopeB = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else "NA"
        if slopeA != "NA" and slopeB != "NA":
            if slopeA != slopeB:
                jun_x = (y3 - y1 + slopeA * x1 - slopeB * x3) / (slopeA - slopeB)
                jun_y = slopeA * (jun_x - x1) + y1
                if inBetween(jun_x, *minMax(x1, x2)) and inBetween(jun_y, *minMax(y1, y2)) \
                and inBetween(jun_x, *minMax(x3, x4)) and inBetween(jun_y, *minMax(y3, y4)):
                        return True
            else:
                jun_y = (x3 - x1) * slopeA + y1
                if abs(jun_y - y3) < 0.0000001 and \
                (inBetween(x1, *minMax(x3, x4)) or inBetween(x2, *minMax(x3, x4)) \
                 or inBetween(x3, *minMax(x1, x2)) or inBetween(x4, *minMax(x1, x2))):
                    return True
        elif slopeA == "NA" and slopeB != "NA":
            jun_x = x1
            jun_y = slopeB * (jun_x - x3) + y3
            if inBetween(jun_x, *minMax(x3, x4)) and inBetween(jun_y, *minMax(y3, y4)) \
            and inBetween(jun_y, *minMax(y1, y2)):
                    return True
        elif slopeA != "NA" and slopeB == "NA":
            jun_x = x3
            jun_y = slopeA * (jun_x - x1) + y1
            if inBetween(jun_x, *minMax(x1, x2)) and inBetween(jun_y, *minMax(y1, y2)) \
            and inBetween(jun_y, *minMax(y3, y4)):
                    return True
        else:
            if x1 == x3:
                return True
        return False
    # check within
    if inBetween(line[0][0], xmin, xmax) and inBetween(line[1][0], xmin, xmax) \
    and inBetween(line[0][1], ymin, ymax) and inBetween(line[1][1], ymin, ymax):
            return True
    # check intersect
    elif isIntersect(line, bBot) or isIntersect(line, bLeft) \
    or isIntersect(line, bRight) or isIntersect(line, bTop):
        return True
    # otherwise outside
    else:
        return False

def getDirChangeAngle(rid, nRid, roads, rDir='pre'):
    # use rid end piece as dir vector
    # compute cumulaitve angle change for each piece of nRid if it have mutiple piece
    if rDir == 'pre':
        rid_pts = np.array(roads[rid].shape_pts[:2])
    elif rDir == 'next':
        rid_pts = np.array(roads[rid].shape_pts[-2:])
    else:
        raise Exception("unrecognized direction %s" % dir)
    rid_dir = rid_pts[1] - rid_pts[0]
    nRid_pts = np.array(roads[nRid].shape_pts)
    nRid_dir = nRid_pts[-1] - nRid_pts[0]
    angle_change = np.arccos(1-distance.cosine(rid_dir, nRid_dir))
    return angle_change

def getPossibleNeighbors(rid, linkRoadLength, pdir, roads, cumuDirChange=0., prePath=[]):
    # idea: some link-like segment representing intersection can be ignored while propagation
    # the cumulative direction change compare to original starting segment can only be less than 180 degree
    # this is to avoid U-turn like congestion propagation for main roads, which is very unlikely
    if pdir == 'pre':
        neighbors = set(roads[rid].preRids)
    elif pdir == 'next':
        neighbors = set(roads[rid].nextRids)
    else:
        raise ValueError("Unknown direction arguement %s" % pdir)
    returnNextRids = {} # {possible reachable neighbor id: [path to it]}
    prePath = [rid] if len(prePath) == 0 else prePath
    for nRid in neighbors:
        sp = copy.copy(prePath)
        sp.append(nRid)
        newCumuDirChange = cumuDirChange + getDirChangeAngle(rid, nRid, roads, pdir)
        # to account for cumulative error caused by acos and not perfect parallel road in opposite dir
        if newCumuDirChange < int(math.pi) + 0.1: 
            if roads[nRid].length <= linkRoadLength:
                returnNextRids.update(getPossibleNeighbors(nRid, linkRoadLength, pdir, roads, 
                                                           cumuDirChange = newCumuDirChange, prePath=sp))
            else:
                returnNextRids[nRid] = sp
    return returnNextRids #{rid: [path_to_it]}

def isUTurn(rid, nRid, roads, rDir):
    if rDir == 'pre':
        rid_pts = np.array(roads[rid].shape_pts[:2])
        nRid_pts = np.array(roads[nRid].shape_pts[-2:])
    elif rDir == 'next':
        rid_pts = np.array(roads[rid].shape_pts[-2:])
        nRid_pts = np.array(roads[nRid].shape_pts[:2])
    else:
        raise Exception("unrecognized direction %s" % dir)
    rid_dir = rid_pts[1] - rid_pts[0]
    nRid_dir = nRid_pts[1] - nRid_pts[0]
    angle_change = np.arccos(1-distance.cosine(rid_dir, nRid_dir))
    if angle_change >= math.pi - 0.001:
        return True
    else:
        return False
    
def readGeojson(fpath, gridLen, selectedRoadTypes=[], selectedRoadIDs=[], noUTurn=False):
    with open(fpath, 'r') as rd:
        road_json = json.loads(rd.read())
    # assign road segments to grids based on intersection
    roads = {}
    grid_to_road = {}
    road_to_grid = {}
    segments_json = road_json["features"]
    roadTypeSel = False
    roadIDSel = False
    if len(selectedRoadTypes) == 0 and len(selectedRoadIDs) == 0:
        RoadSelection=False
    else:
        RoadSelection=True
        removeRidSet = set()
        if len(selectedRoadTypes) > 0:
            roadTypeSel = True
            roadTypes = set(selectedRoadTypes)
        if len(selectedRoadIDs) > 0:
            roadIDSel = True
            roadIDs = set(selectedRoadIDs)
    def str2IntList(l):
        l = l.split(',') if len(l) > 0 else []
        return [int(s) for s in l]
    print 'original # of roads: %d' % len(segments_json)
    for segment_json in segments_json:
        rid = segment_json["id"]
        roadType = segment_json["properties"]["roadType"]
        recordRoad = True
        if roadTypeSel and (roadType not in roadTypes):
            recordRoad = False
        if roadIDSel and (rid not in roadIDs):
            recordRoad = False
        if recordRoad:
            nextRids = segment_json["properties"]["nextAdj"]
            preRids = segment_json["properties"]["preAdj"]
            nextRids = str2IntList(nextRids)
            preRids = str2IntList(preRids)
            shape_pts = segment_json["geometry"]["coordinates"]
            roads[rid] = Road(rid, nextRids, preRids, shape_pts, roadType)
        else:
            removeRidSet.add(rid)    
    if RoadSelection:
        # clear neighboring info if road selection is enabled
        for rid in roads:
            roads[rid].preRids = list(set(roads[rid].preRids) - removeRidSet)
            roads[rid].nextRids = list(set(roads[rid].nextRids) - removeRidSet)
    if noUTurn:
        print 'U-Turn is ignored in network connectivity'
        for rid in roads:
            preRemove = set()
            nextRemove = set()
            for nRid in roads[rid].preRids:
                if isUTurn(rid, nRid, roads, 'pre'):
                       preRemove.add(nRid)
            for nRid in roads[rid].nextRids:
                if isUTurn(rid, nRid, roads, 'next'):
                       nextRemove.add(nRid)
            roads[rid].preRids = list(set(roads[rid].preRids) - preRemove)
            roads[rid].nextRids = list(set(roads[rid].nextRids) - nextRemove)
    else:
        print 'U-Turn is accounted in network connectivity'
    # determine NO. of grids in X, Y
    # code will be 2-digits, starting from 00 to XY
    xmin, ymin, xmax, ymax = road_json["bbox"]
    x_nGrids = math.ceil((xmax - xmin)/gridLen)
    y_nGrids = math.ceil((ymax - ymin)/gridLen)
    print 'NO. X-grids: %d, NO. Y-grids: %d' % (x_nGrids, y_nGrids)           
    # hash roads to grids
    for rid in roads:
        shape_pts = roads[rid].shape_pts
        # determine which grids the road segment intersects with
        for k in xrange(len(shape_pts)-1):
            stPt = shape_pts[k]
            edPt = shape_pts[k+1]
            line_seg = [stPt, edPt]
            st_nX, st_nY = getGridNumForPoint(stPt, xmin, ymax, gridLen)
            ed_nX, ed_nY = getGridNumForPoint(edPt, xmin, ymax, gridLen)
            # check every grid in the bounding grid box of this line segment, see which it intersects with
            nX_min, nX_max, nY_min, nY_max = min(st_nX, ed_nX), max(st_nX, ed_nX), min(st_nY, ed_nY), max(st_nY, ed_nY)
            for nx in xrange(nX_min, nX_max + 1):
                for ny in xrange(nY_min, nY_max + 1):
                    grid_bbox = [xmin + nx * gridLen, ymax - (ny + 1) * gridLen, xmin + (nx + 1) * gridLen, ymax - ny * gridLen]
                    grid_id = (nx, ny)
                    if _isLineOverlapSquare(line_seg, grid_bbox):
                        _addToListDict(grid_to_road, grid_id, rid)
                        _addToListDict(road_to_grid, rid, grid_id)
    topLeftOrigin = (xmin, ymax)
    xyNGrids = (x_nGrids, y_nGrids)
    return roads, topLeftOrigin, xyNGrids, grid_to_road, road_to_grid

def find_LCC_update_Adj(roads, linkRoadMaxLength = 0):
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
    selectedRids = np.sort(np.array(list(LCS)))
    print "# of roads in the largest connected component: %d" % selectedRids.shape[0]
   
    # reconstruct road network adjacency to exclude those short-length linking roads during propagation
    linkRoads = set()
    if linkRoadMaxLength > 0:
        modAdjPrePath = {}
        modAdjNextPath = {}
        for rid in roads:
            if roads[rid].length > linkRoadMaxLength:
                new_preRids = set()
                for nRid in roads[rid].preRids:
                    if roads[nRid].length <= linkRoadMaxLength:
                        newAdj = getPossibleNeighbors(rid, linkRoadMaxLength, 'pre', roads)
                        for newAdj_rid in newAdj:
                            modAdjPrePath[(rid, newAdj_rid)] = newAdj[newAdj_rid]
                        new_preRids |= set(newAdj.keys())
                    else:
                        new_preRids.add(nRid)
                roads[rid].preRids = list(new_preRids)
        
                new_nextRids = set()
                for nRid in roads[rid].nextRids:
                    if roads[nRid].length <= linkRoadMaxLength:
                        newAdj = getPossibleNeighbors(rid, linkRoadMaxLength, 'next', roads)
                        for newAdj_rid in newAdj:
                            modAdjNextPath[(rid, newAdj_rid)] = newAdj[newAdj_rid]
                        new_nextRids |= set(newAdj.keys())
                    else:
                        new_nextRids.add(nRid)
                roads[rid].nextRids = list(new_nextRids)
            else:
                linkRoads.add(rid)
    print '# of link roads: %d' % len(linkRoads)
    return selectedRids, linkRoads

def DijkstraSP(roads, relatedRids, startRid, direction='undir', nNN=50, endRid=None, weighted=True):
    # if endRid is None, compute shorest path to all roads. If not, compute shortest path to endRid
    # sanity check
    relRids = set(relatedRids)
    if startRid not in relRids:
        raise ValueError("Dijkstra: start road segment %d not in input related roads" % startRid)
    if (endRid != None) and (endRid not in relRids):
        raise ValueError("Dijkstra: start road segment %d not in input related roads" % endRid)
    # in case start gps point and end gps point are projected onto the same road
    current = startRid
    _SPParent = {current:current}
    _SPDistance = {rid: float('inf') for rid in relRids} # shortest distance to segment
    _unvisited = set(relatedRids)
    _SPDistance[current] = 0.
    _nVisitedRids = 0
    while True:
        if direction == 'predir':
            nIter_neighbors = roads[current].preRids
        elif direction == 'nextdir':
            nIter_neighbors = roads[current].nextRids
        else:
            nIter_neighbors = set(roads[current].preRids) | set(roads[current].nextRids)
        for rid in nIter_neighbors:
            if rid in relRids:
                if weighted:
                    dist = _SPDistance[current] + roads[rid].length
                else:
                    dist = _SPDistance[current] + 1.0
                if dist < _SPDistance[rid]:
                    _SPParent[rid] = current
                    _SPDistance[rid] = dist
        _unvisited.remove(current)
        _nVisitedRids += 1
        if endRid == None:
            if (_nVisitedRids == nNN) or all(_SPDistance[rid] == float('inf') for rid in _unvisited):
                break
            else:
                current = min(_unvisited, key = lambda x: _SPDistance[x])
        else:
            if (current == endRid) or all(_SPDistance[rid] == float('inf') for rid in _unvisited):
                break
            else:
                current = min(_unvisited, key = lambda x: _SPDistance[x])
    _visitedRids = set(relatedRids) - _unvisited
    _visitedRids.remove(startRid)
    if weighted:
        for rid in _visitedRids:
            _SPDistance[rid] = _SPDistance[rid] + (roads[startRid].length - roads[rid].length) / 2.0
    if endRid == None:
        return _SPDistance, _visitedRids
    else:
        if current == endRid:
            _SPPath = [endRid]
            rid = endRid
            while True:
                p = _SPParent[rid]
                _SPPath.append(p)
                rid = p
                if rid == startRid:
                    break
            _SPPath = [rid for rid in reversed(_SPPath)]
            return _SPDistance, _SPPath
        else:
            return float('inf'), []