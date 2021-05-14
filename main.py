### changing directory to import modules
# print(os.getcwd())
# os.chdir("Desktop/Rijurekha Sen/interpol/Final/preprocessing/")
# print(os.getcwd())
# print(os.listdir("../../data/"))


# import modules
from Final.models.ApproximateGPR import train_AppxGpr
from Final.models.GCN import train_GCN

# load data
### AppxGPR
df = pd.read_csv("../data/2020-12-24.csv")
train_tuple = x_train, y_train

### GCN
dataset = pd.read_csv('drive/MyDrive/data.csv')
Data1Hr = dataset[dataset['GENERATION_TIME']<'2012.02.01 01:00']
data = Data1Hr[['LATITUDE','LONGITUDE', 'OZONE_PPB']]

X = torch.tensor(data.values[:, :2])
y = torch.tensor(data.values[:, 2])
data_tuple = (X, y)

# preprocess data
### reduce data size 

### normalize 
normalizer = True
if normalizer:
    train_tuple, test_tuple, x_ms, y_ms = normalize(train_tuple, test_tuple=None)
    train_x, train_y = train_tuple
else:
    x_ms = (0, 1)
    y_ms = (0, 1)

ms = (x_ms, y_ms)  

### Making a graph
normalizeData = (data - data.mean(axis = 0)) / data.std(axis = 0)
npData = normalizeData.to_numpy(dtype = float)

train_create = False

if train_create:
  trainRatio = 0.95
else: 
  trainRatio = 1

trainMask = random.sample(list(range(npData.shape[0])),int(npData.shape[0]*trainRatio))
trainMask.sort()  
print(len(trainMask))

data_tuple, _, x_ms, y_ms = normalize(data_tuple)


edges = make_edges(data_tuple)
graph = make_graph(data_tuple, trainMask, edges)

# run models

### ApproximateGPR
AppxGpr = train_AppxGpr(train_tuple, y_ms, mean=0, kernel=0, ard = True, \
                        num_epochs = 100, ind_pts = 500, cuda = False)

### GCN
train_GCN(graph, y_ms, num_epochs = 100, cuda = False)    
    
# save trained models


# train and test results 