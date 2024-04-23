print("hello")

# Import the required modules
import numpy as np
from MT.refinement import train_and_evaluate_model_blind, train_and_evaluate_model_SRR_light
import os

# Define path
save_folder = "./results/Patchcore/"
name = "20240419_Patchcore_all_categories_cont_max_0.15_runs_05_sampling_ratio_0.01_no_refinement.npy"
name_simple = "20240419_Patchcore_all_categories_cont_max_0.15_runs_05_sampling_ratio_0.01_simple_refinement.npy"
name_SRR = "20240419_Patchcore_all_categories_cont_max_0.15_runs_05_sampling_ratio_0.01_SRR_refinement.npy"
file_path_blind = os.path.join(save_folder, name)
file_path_simple = os.path.join(save_folder, name_simple)
file_path_SRR = os.path.join(save_folder, name_SRR)


#Define experiment
run_arr = np.array([1, 2, 3, 4, 5]) #np.array([1, 2]) #
cont_ratio_arr = np.array([0.0, 0.05, 0.1, 0.15]) #np.array([0.15]) # 
category_arr = np.array(["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"])#np.array(["carpet", "grid"])#np.array(["metal_nut"])#, "grid"])# #
# np.array(["cable", "wood", "metal_nut"]) #
# Initialize results arrays
results_blind_arr = np.empty([run_arr.shape[0], cont_ratio_arr.shape[0], category_arr.shape[0]])
results_simple_arr = np.empty([run_arr.shape[0], cont_ratio_arr.shape[0], category_arr.shape[0]])
results_SRR_arr = np.empty([run_arr.shape[0], cont_ratio_arr.shape[0], category_arr.shape[0]])


for idx_run, run in enumerate(run_arr):
    for idx_cont_ratio, cont_ratio in enumerate(cont_ratio_arr):
        for idx_category, category in enumerate(category_arr):

            print("+-----------------------+")
            print("| run: ", run, "         |")
            print("| cont_ratio: ", cont_ratio, "   |")
            print("| category: ", category, "   |")
            print("+-----------------------+")

            # Define refinement
            coreset_sampling_ratio = 0.01
            gamma = 1-cont_ratio
            k = 5

            # Evaluate Blind
            results_test_blind = train_and_evaluate_model_blind(coreset_sampling_ratio, run, category, cont_ratio)
            results_blind_arr[idx_run, idx_cont_ratio, idx_category] = results_test_blind[0]["image_AUROC"]

            # Evaluate Simple Refine
            results_test_simple, _, _, _, _, _ = train_and_evaluate_model_SRR_light(1, gamma, coreset_sampling_ratio, run, category, cont_ratio)
            results_simple_arr[idx_run, idx_cont_ratio, idx_category] = results_test_simple[0]["image_AUROC"]

            # Evaluate SRR Light
            results_test_SRR, _, _, _, _, _ = train_and_evaluate_model_SRR_light(k, gamma, coreset_sampling_ratio, run, category, cont_ratio)
            results_SRR_arr[idx_run, idx_cont_ratio, idx_category] = results_test_SRR[0]["image_AUROC"]
            

# Save results_arr to the specified folder
np.save(file_path_blind, results_blind_arr)
np.save(file_path_simple, results_simple_arr)
np.save(file_path_SRR, results_SRR_arr)



