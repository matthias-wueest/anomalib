

# Import the required modules
import numpy as np
from lightning.pytorch import seed_everything
from anomalib.data.image.mvtec import MVTec_contaminated, MVTecDataset_contaminated
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib import TaskType
from anomalib.data.utils import Split


def train_and_evaluate_model_blind(coreset_sampling_ratio, run, category, cont_ratio):
    seed_everything(run, workers=True)
    datamodule = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run, idx=[])
    model = Patchcore(coreset_sampling_ratio=coreset_sampling_ratio)
    engine = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10, devices=1)
    engine.fit(datamodule=datamodule, model=model)

    results_test_blind = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
        verbose=False
    )
    
    return results_test_blind



def train_and_evaluate_model_SRR_light(k, gamma, coreset_sampling_ratio, run, category, cont_ratio):
    print("k: ", k)

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
    print("train_dataset_length:", train_dataset_length)
    indices = np.arange(0, train_dataset_length)
    np.random.seed(run)
    np.random.shuffle(indices)
    indices_disjoint_datasets = np.array_split(indices, k)
    print("indices_disjoint_datasets: ", indices_disjoint_datasets)
    # Train k models on k disjoint datasets
    classifications_subset_arr = np.empty([train_dataset_length,k], dtype=bool)
    for k_iter in range(k):
        print("k_iter: ", k_iter)
        # Train model on disjoint dataset
        seed_everything(run, workers=True)
        datamodule = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run, idx=indices_disjoint_datasets[k_iter])
        model = Patchcore(coreset_sampling_ratio=coreset_sampling_ratio) 
        engine = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10, devices=1)
        engine.fit(datamodule=datamodule, model=model)

        # Predict binary labels for each sample
        predictions_subset = engine.predict(model=model, dataset=train_dataset)
        prediction_scores_subset = np.array([d["pred_scores"][0] for d in predictions_subset])
        print("prediction_scores_subset: ", prediction_scores_subset)
        print("gamma:", gamma)
        threshold = np.percentile(prediction_scores_subset, q=gamma*100)
        print("threshold: ", threshold)
        classifications_subset = prediction_scores_subset>threshold # True: abnormal; False: normal
        print("classifications_subset: ", classifications_subset)
        # Save binary classifications
        classifications_subset_arr[:,k_iter] = classifications_subset


    # Return indices of refined dataset
    keep_bool_arr = np.all(~classifications_subset_arr, axis=1)
    keep_indices = np.where(keep_bool_arr)[0]
    print("keep_indices: ", keep_indices)

    # Evaluate correctness of pseudo-labels
    abnormal_total = 0
    abnormal_excluded = 0
    normal_total = 0
    normal_excluded = 0

    for i in range(train_dataset.__len__()):

        if train_dataset.__getitem__(i)["label"] == 1:
            abnormal_total += 1
            if ~np.isin(i, keep_indices):
                abnormal_excluded += 1
        elif train_dataset.__getitem__(i)["label"] == 0:
            normal_total += 1
            if ~np.isin(i, keep_indices):
                normal_excluded += 1
    

    ## Train and evaluate final model on refined dataset
    # Train one model on refined dataset
    seed_everything(run, workers=True)
    datamodule = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run, idx=keep_indices)
    model = Patchcore(coreset_sampling_ratio=coreset_sampling_ratio)
    engine = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10, devices=1)         
    engine.fit(datamodule=datamodule, model=model)


    # Evaluate model on test set
    #predictions_test_refined = engine_refined.predict(model=model_refined, dataset=datamodule_refined.get_test_dataset())
    results_test_refined = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
        verbose=False
    )


    return results_test_refined, keep_indices, abnormal_total, abnormal_excluded, normal_total, normal_excluded











def train_and_evaluate_model_SRR_light_bkp(k, gamma, coreset_sampling_ratio, run, category, cont_ratio):

    print("k: ", k)

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
    indices = np.arange(0, train_dataset_length)
    np.random.seed(run)
    np.random.shuffle(indices)
    indices_disjoint_datasets = np.array_split(indices, k)

    # Train k models on k disjoint datasets
    classifications_subset_arr = np.empty([train_dataset_length,k], dtype=bool)
    for k_iter in range(k):
        print("k_iter: ", k_iter)
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

    # Return indices of refined dataset
    keep_bool_arr = np.all(~classifications_subset_arr, axis=1)
    keep_indices = np.where(keep_bool_arr)[0]


    # Evaluate correctness of pseudo-labels
    abnormal_total = 0
    abnormal_excluded = 0
    normal_total = 0
    normal_excluded = 0

    for i in range(train_dataset.__len__()):

        if train_dataset.__getitem__(i)["label"] == 1:
            abnormal_total += 1
            if ~np.isin(i, keep_indices):
                abnormal_excluded += 1
        elif train_dataset.__getitem__(i)["label"] == 0:
            normal_total += 1
            if ~np.isin(i, keep_indices):
                normal_excluded += 1


    ## Train and evaluate final model on refined dataset
    # Train one model on refined dataset
    seed_everything(run, workers=True)
    datamodule = MVTec_contaminated(category=category, cont_ratio=cont_ratio, run=run, idx=keep_indices)
    model = Patchcore(coreset_sampling_ratio=coreset_sampling_ratio)
    engine = Engine(task=TaskType.CLASSIFICATION, image_metrics=["AUROC", "AUPR", "F1Score"], max_epochs=10, devices=1)
    engine.fit(datamodule=datamodule, model=model)

    # Evaluate model on test set
    results_test_refined = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
        verbose=False
    )
    return results_test_refined, keep_indices, abnormal_total, abnormal_excluded, normal_total, normal_excluded
