## tested on python2
## Map match trajectories
# Think - to keep delta small, we might not want to remove redundant points

from fmm import FastMapMatch,Network,NetworkGraph,UBODTGenAlgorithm,UBODT,FastMapMatchConfig,STMATCH,STMATCHConfig
from fmm import GPSConfig,ResultConfig
import glob
import pickle

### Read network data
map_input_dir = "mapdata/"
edges_fname = map_input_dir + "edges.shp"
# ubodt_fname = map_input_dir + "ubodt_0.06_Oct25.txt"
# ubodt_fname = map_input_dir + "ubodt_0.06_Nov14.txt"
ubodt_fname = map_input_dir + "ubodt_0.06.txt"
network = Network(edges_fname,"fid","u","v")
print("Nodes {} edges {}".format(network.get_node_count(),network.get_edge_count()))
graph = NetworkGraph(network)

# ### Precompute an UBODT table
# Can be skipped if you already generated an ubodt file
ubodt_gen = UBODTGenAlgorithm(network,graph)
status = ubodt_gen.generate_ubodt(ubodt_fname, delta=0.06, binary=False, use_omp=False)
print(status)
### Read UBODT
ubodt = UBODT.read_ubodt_csv(ubodt_fname)

### Create FMM model
model = FastMapMatch(network,graph,ubodt)
### Define map matching configuration
k = 8
radius = 0.05
gps_error = 0.05
fmm_config = FastMapMatchConfig(k,radius,gps_error)

### Run map matching from csv file of trips
# https://github.com/cyang-kth/fmm/blob/master/example/notebook/fmm_example.ipynb

def construct_string(traj):
    s = 'LINESTRING('
    for (x,y) in traj:
        s += '{} {},'.format(y,x)
    s = s[:-1] + ')'
    return s

f = open('wkt.pkl', 'rb')
trajs = pickle.load(f)
f.close()

# wkts will be a list of trajs to match
# you also have to construct the Linestring
print('Have to match {} trips'.format(len(trajs)))

outputs = []
for traj in trajs:
    wkt = construct_string(traj)
    result = model.match_wkt(wkt, fmm_config)
    candidates = []
    for c in result.candidates:
        candidates.append((c.edge_id,c.source,c.target,c.error,c.length,c.offset))
    print('to match: {}, matched: {}'.format(len(traj),len(candidates)))
    outputs.append((candidates, list(result.cpath)))

f = open('wkt_mm.pkl', 'wb')
pickle.dump(outputs, f)
f.close()
