# Import the required modules
import numpy as np
from lightning.pytorch import seed_everything
from anomalib.data.image.mvtec import MVTec_contaminated, MVTecDataset_contaminated
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib import TaskType
from anomalib.data.utils import Split


coreset_sampling_ratio = 0.01
run = 1
category = "cable"
cont_ratio = 0.15


seed_everything(run, workers=True)
train_dataset = MVTecDataset_contaminated(
                task=TaskType.CLASSIFICATION,
                split=Split.TRAIN,
                category=category,
                cont_ratio=cont_ratio,
                run=run,
                idx = []           
            )

# Create indices for k disjoint datasets
train_dataset_length = train_dataset.__len__()


# Input
N = train_dataset_length#23
M = 8
M_train = 4

ones = np.ones(M_train, dtype=bool)
zeros = np.zeros(M-M_train, dtype=bool)
period_bool = np.hstack((ones, zeros))
#print(period_bool)

subset1_bool = np.array([], dtype=bool)
#for i in range(1,int(np.ceil(N/M))):
for i in range(int(np.ceil(N/M))):
    subset1_bool = np.hstack((subset1_bool, period_bool))
#print(subset1_bool)

subsets_bool = subset1_bool
for m in range(1,M):
    #print(m)
    subset_new_bool = np.roll(subset1_bool, shift=m)
    subsets_bool = np.vstack((subsets_bool, subset_new_bool))

#subsets_bool = subsets_bool[:,:-(M-np.mod(N, M))]

subsets_bool = subsets_bool[:,:int(np.floor(N/M)*M + np.mod(N, M))]
print(subsets_bool.astype(int))



indices = np.arange(0, train_dataset_length)
np.random.seed(run)
np.random.shuffle(indices)


# Train M models on M partially overlapping datasets
m_iter = 0
indices_subset = indices[np.where(subsets_bool.astype(int)[m_iter])]
indices_subset

seed_everything(run, workers=True)
datamodule = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run, idx=indices_subset)
model = Patchcore(coreset_sampling_ratio=coreset_sampling_ratio) 
engine = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10, devices=1)
engine.fit(datamodule=datamodule, model=model)

# Predict binary labels for each sample
predictions_subset = engine.predict(model=model, dataset=train_dataset)