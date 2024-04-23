print("hello")

import numpy as np
import os
from pathlib import Path

# Import the required modules
from lightning.pytorch import Trainer, seed_everything
from anomalib.data import MVTec
from anomalib.data.image.mvtec import MVTec_contaminated
from anomalib.models import Padim, Patchcore, Stfpm, Draem, EfficientAd
from anomalib.engine import Engine
from anomalib import TaskType


# Define path
save_folder = "./results/Patchcore/"
name = "20240417_Patchcore_all_categories_cont_max_0.15_runs_05_sampling_ratio_0.01_no_refinement.npy"
name_refined = "20240417_Patchcore_all_categories_cont_max_0.15_runs_05_sampling_ratio_0.01_simple_refinement.npy"
file_path = os.path.join(save_folder, name)
file_path_refined = os.path.join(save_folder, name_refined)

#Define experiment
run_arr = np.array([1, 2, 3, 4, 5]) #np.array([1]) #np.array([42])#np.arange(3)+1
cont_ratio_arr = np.array([0.0, 0.05, 0.1, 0.15]) #np.array([0.0, 0.15])#np.array([0.15])#, 0.1]) #, 0.15
#category_arr = np.array(["cable", "metal_nut", "transistor"])
category_arr = np.array(["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"])#np.array(["carpet", "grid"])#np.array(["metal_nut"])#, "grid"])# #
results_arr = np.empty([run_arr.shape[0], cont_ratio_arr.shape[0], category_arr.shape[0]])
results_arr_refined = np.empty([run_arr.shape[0], cont_ratio_arr.shape[0], category_arr.shape[0]])

for idx_run, run in enumerate(run_arr):
    for idx_cont_ratio, cont_ratio in enumerate(cont_ratio_arr):
        for idx_category, category in enumerate(category_arr):

            # Define seed, datamodule, model and engine
            seed_everything(run, workers=True)
            datamodule = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run, idx=[])
            model = Patchcore(coreset_sampling_ratio=0.01) #Padim(backbone="resnet18", n_features=100, layers=["layer1", "layer2", "layer3"])
            engine = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10, devices=1)

            # Train model on training set
            engine.fit(datamodule=datamodule, model=model)

            # Evaluate model on test set
            test_results = engine.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
                verbose=False
            )
            results_arr[idx_run, idx_cont_ratio, idx_category] = test_results[0]["image_AUROC"]
            
            # Refine training set 
            prediction_dataset = datamodule.get_train_dataset()
            predictions = engine.predict(model=model, dataset=prediction_dataset) # Make predictions on training set
            prediction_scores = np.array([d["pred_scores"][0] for d in predictions]).tolist() # Get list of prediction scores
            sorted_indices = sorted(range(len(prediction_scores)), key=lambda i: prediction_scores[i]) # Sort the training samples based on prediction scores
            selected_indices = sorted_indices[:int((1-cont_ratio) * len(sorted_indices))] # Select the (1-cont_ratio)*100% lowest prediction score samples

            
            # Define seed, datamodule, model and engine
            seed_everything(run, workers=True)
            datamodule_refined = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run, idx=selected_indices)
            model_refined = Patchcore(coreset_sampling_ratio=0.01)
            engine_refined = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10, devices=1)

            # Train the model
            engine_refined.fit(datamodule=datamodule_refined, model=model_refined)

            # Load best model from checkpoint before evaluating
            test_results_refined = engine_refined.test(
                model=model_refined,
                datamodule=datamodule_refined,
                ckpt_path=engine_refined.trainer.checkpoint_callback.best_model_path,
                verbose=False
            )
            results_arr_refined[idx_run, idx_cont_ratio, idx_category] = test_results_refined[0]["image_AUROC"]
            
            print(run)
            print(cont_ratio)
            print(category)

# Save results_arr to the specified folder
np.save(file_path, results_arr)
np.save(file_path_refined, results_arr_refined)