print("hello")

# Import the required modules
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import Trainer, seed_everything
from anomalib.data import MVTec
from anomalib.data.image.mvtec import MVTec_contaminated, MVTecDataset_contaminated, make_mvtec_dataset, make_mvtec_dataset_contaminated
from anomalib.models import Padim, Patchcore, Stfpm, Draem, EfficientAd
from anomalib.engine import Engine
from anomalib import TaskType

from anomalib.data.utils import Split

# Define experiment
run_arr = np.array([1, 2, 3, 4, 5]) # np.array([1, 2])#, 2])#
#category_arr = np.array(["cable", "wood", "metal_nut"])#np.array(["cable"])#
category_arr = np.array(["capsule", "screw", "toothbrush"])

# Initialize arrays for saving
results_test_blind_arr = np.empty([run_arr.shape[0], category_arr.shape[0]], dtype=object)
results_test_refined_arr = np.empty([run_arr.shape[0], category_arr.shape[0]], dtype=object)
keep_indices_arr = np.empty([run_arr.shape[0], category_arr.shape[0]], dtype=object)
train_dataset_length_arr = np.empty([run_arr.shape[0], category_arr.shape[0]], dtype=object)
abnormal_total_arr = np.empty([run_arr.shape[0], category_arr.shape[0]], dtype=object)
abnormal_excluded_arr = np.empty([run_arr.shape[0], category_arr.shape[0]], dtype=object)
normal_total_arr = np.empty([run_arr.shape[0], category_arr.shape[0]], dtype=object)
normal_excluded_arr = np.empty([run_arr.shape[0], category_arr.shape[0]], dtype=object)
k_arr = np.empty([run_arr.shape[0], category_arr.shape[0]], dtype=object)


# Link: https://pytorch.org/blog/understanding-gpu-memory-1/ 
import logging
logger = logging.getLogger(__name__)
torch.cuda.memory._record_memory_history(
    max_entries=100000
) 

# Loop over experiments
for idx_category, category in enumerate(category_arr):
    for idx_run, run in enumerate(run_arr):
        
        print("------------------------------------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------------------------------------")
        print("category: ", category)
        print("run: ", run)
        print("------------------------------------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------------------------------------")

        ## Loop
        cont_ratio = 0.15
        coreset_sampling_ratio = 0.01

        ## Define refinement
        #k_ls = [1, 2, 3, 5, 8] #k_arr = np.array([1, 2, 4, 8]) #np.array([1, 2, 3, 4, 5, 8, 10, 15])
        #k_ls = [1]
        k_ls = [1, 2, 3, 5, 10]
        #k_ls = [1, 2, 3, 5, 10, 20, 40]
        gamma = 1-cont_ratio

        ## Lists to save results
        results_test_blind_ls=[]
        results_test_refined_ls=[]
        keep_indices_ls=[]
        train_dataset_length_ls=[]
        abnormal_total_ls=[]
        abnormal_excluded_ls=[]
        normal_total_ls=[]
        normal_excluded_ls=[]


        ## Evaluate baseline (blind training)
        # Train one model on refined dataset
        seed_everything(run, workers=True)
        datamodule = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run, idx=[])
        model = Patchcore(coreset_sampling_ratio=coreset_sampling_ratio)
        engine = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10, devices=1)
        engine.fit(datamodule=datamodule, model=model)

        # Evaluate model on test set
        results_test_blind = engine.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
            verbose=False
        )
        
        train_dataset = MVTecDataset_contaminated(
                    task=TaskType.CLASSIFICATION,
                    split=Split.TRAIN,
                    category=category,
                    cont_ratio=cont_ratio,
                    run=run,
                    idx = []           
                )
        train_dataset_length = train_dataset.__len__()
        indices = np.arange(0, train_dataset_length)
        np.random.seed(run)
        np.random.shuffle(indices)

        for k in k_ls:
            #torch.cuda.empty_cache()
            # Create indices for k disjoint datasets

            print("------------------------------------------------------------------------------------------------------------------------")
            print("------------------------------------------------------------------------------------------------------------------------")
            print("category: ", category)
            print("run: ", run)
            print("k: ", k)
            print("------------------------------------------------------------------------------------------------------------------------")
            print("------------------------------------------------------------------------------------------------------------------------")
            
            indices_disjoint_datasets = np.array_split(indices, k)
            #torch.cuda.empty_cache()

            # Train k models on k disjoint datasets
            classifications_subset_arr = np.empty([train_dataset_length,k], dtype=bool)
            for k_iter in range(k):
                print("------------------------------------------------------------------------------------------------------------------------")
                print("------------------------------------------------------------------------------------------------------------------------")
                print("category: ", category)
                print("run: ", run)
                print("k: ", k)
                print("k_iter: ", k_iter)
                print("------------------------------------------------------------------------------------------------------------------------")
                print("------------------------------------------------------------------------------------------------------------------------")
                #torch.cuda.empty_cache()
                # Train model on disjoint dataset
                seed_everything(run, workers=True)
                datamodule = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run, idx=indices_disjoint_datasets[k_iter])
                model = Patchcore(coreset_sampling_ratio=coreset_sampling_ratio) 
                engine = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10, devices=1)
                engine.fit(datamodule=datamodule, model=model)

                # Predict binary labels for each sample
                predictions_subset = engine.predict(model=model, dataset=train_dataset)
                prediction_scores_subset = np.array([d["pred_scores"][0] for d in predictions_subset])
                threshold = np.percentile(prediction_scores_subset, q=gamma*100)
                classifications_subset = prediction_scores_subset>threshold # True: abnormal; False: normal

                # Save binary classifications
                classifications_subset_arr[:,k_iter] = classifications_subset
                #torch.cuda.empty_cache()

            # Return indices of refined dataset
            keep_bool_arr = np.all(~classifications_subset_arr, axis=1)
            keep_indices = np.where(keep_bool_arr)[0]


            # Evaluate correctness of pseudo-labels
            abnormal_total = 0
            abnormal_excluded = 0
            normal_total = 0
            normal_excluded = 0
            #torch.cuda.empty_cache()
            for i in range(train_dataset.__len__()):

                if train_dataset.__getitem__(i)["label"] == 1:
                    abnormal_total += 1
                    if ~np.isin(i, keep_indices):
                        abnormal_excluded += 1
                elif train_dataset.__getitem__(i)["label"] == 0:
                    normal_total += 1
                    if ~np.isin(i, keep_indices):
                        normal_excluded += 1
            
            try:
                torch.cuda.memory._dump_snapshot("snapshot.pickle")
            except Exception as e:
                logger.error(f"Failed to capture memory snapshot {e}")

            ## Train and evaluate final model on refined dataset
            #torch.cuda.empty_cache()
            # Train one model on refined dataset
            seed_everything(run, workers=True)
            datamodule = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run, idx=keep_indices)
            model = Patchcore(coreset_sampling_ratio=coreset_sampling_ratio)
            engine = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10, devices=1)
            try:
                torch.cuda.memory._dump_snapshot("snapshot.pickle")
            except Exception as e:
                logger.error(f"Failed to capture memory snapshot {e}")            
            engine.fit(datamodule=datamodule, model=model)


            # Evaluate model on test set
            #predictions_test_refined = engine_refined.predict(model=model_refined, dataset=datamodule_refined.get_test_dataset())
            results_test_refined = engine.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
                verbose=False
            )
            #torch.cuda.empty_cache()

            ## Save results
            results_test_blind_ls.append(results_test_blind)
            results_test_refined_ls.append(results_test_refined)
            keep_indices_ls.append(keep_indices)
            train_dataset_length_ls.append(train_dataset_length)
            abnormal_total_ls.append(abnormal_total)
            abnormal_excluded_ls.append(abnormal_excluded)
            normal_total_ls.append(normal_total)
            normal_excluded_ls.append(normal_excluded)

            try:
                torch.cuda.memory._dump_snapshot("snapshot.pickle")
            except Exception as e:
                logger.error(f"Failed to capture memory snapshot {e}")

        ## Save results in array
        #torch.cuda.empty_cache()
        results_test_blind_arr[idx_run, idx_category] = results_test_blind_ls
        results_test_refined_arr[idx_run, idx_category] = results_test_refined_ls
        keep_indices_arr[idx_run, idx_category] = keep_indices_ls
        train_dataset_length_arr[idx_run, idx_category] = train_dataset_length_ls
        abnormal_total_arr[idx_run, idx_category] = abnormal_total_ls
        abnormal_excluded_arr[idx_run, idx_category] = abnormal_excluded_ls
        normal_total_arr[idx_run, idx_category] = normal_total_ls
        normal_excluded_arr[idx_run, idx_category] = normal_excluded_ls
        k_arr[idx_run, idx_category] = k_ls

        #torch.cuda.empty_cache()

## Save arrays in dict
results_dict = {
    "dimensions": {
        "run_arr": run_arr,
        "category_arr": category_arr
        
    },
    "results": {
        "results_test_blind_arr": results_test_blind_arr,
        "results_test_refined_arr": results_test_refined_arr,
        "keep_indices_arr": keep_indices_arr,
        "train_dataset_length_arr": train_dataset_length_arr,
        "abnormal_total_arr": abnormal_total_arr,
        "abnormal_excluded_arr": abnormal_excluded_arr,
        "normal_total_arr": normal_total_arr,
        "normal_excluded_arr":   normal_excluded_arr,
        "k_arr": k_arr
    }
}

# Stop recording memory snapshot history.
torch.cuda.memory._record_memory_history(enabled=None)



# Define path
save_folder = "./results/Patchcore/"
#name = "20240417_Patchcore_SRR_cable_run12345_k_1_40.pickle"
name = "20240423_Patchcore_SRR_capsule_screw_toothbrush_run12345_k_1_10.pickle"
target_path = os.path.join(save_folder, name)

# Save results
import pickle
with open(target_path, "wb") as pickle_file:
    pickle.dump(results_dict, pickle_file)
