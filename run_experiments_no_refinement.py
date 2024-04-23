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


save_folder = "./results/Patchcore/"#"./results/Stfpm/" # 
name = "Patchcore_all_categories_cont_max_0.15_runs_05_sampling_ratio_0.01_20240404.npy" #"Patchcore_all_categories_cont_max_0.15_runs_10_20240331.npy"#"Stfpm_20240328.npy" # 
file_path = os.path.join(save_folder, name)
file_path

run_arr = np.array([1, 2, 3, 4, 5])#np.array([1]) #np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #np.array([1]) #np.array([1]) #np.array([42])#np.arange(3)+1
cont_ratio_arr = np.array([0.0, 0.05, 0.1, 0.15]) #np.array([0.0])#np.array([0.0])#np.array([0.0, 0.05, 0.1]) #, 0.15
category_arr = np.array(["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"])#np.array(["carpet"])#np.array(["carpet"])#np.array(["carpet", "grid"]) #
results_arr = np.empty([run_arr.shape[0], cont_ratio_arr.shape[0], category_arr.shape[0]])

for idx_run, run in enumerate(run_arr):
    for idx_cont_ratio, cont_ratio in enumerate(cont_ratio_arr):
        for idx_category, category in enumerate(category_arr):

            seed_everything(run, workers=True)
            datamodule = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run)
            #model = Padim(backbone="resnet18", n_features=100, layers=["layer1", "layer2", "layer3"])
            #model = EfficientAd()
            model = Patchcore(coreset_sampling_ratio=0.01)
            engine = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10)

            # Train the model
            engine.fit(datamodule=datamodule, model=model)

            # load best model from checkpoint before evaluating
            test_results = engine.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
                verbose=False
            )
            test_results
            results_arr[idx_run, idx_cont_ratio, idx_category] = test_results[0]["image_AUROC"]
            
            print(run)
            print(cont_ratio)
            print(category)

# Save results_arr to the specified folder
np.save(file_path, results_arr)
print("done")