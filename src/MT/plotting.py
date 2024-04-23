
import numpy as np
import matplotlib.pyplot as plt

def plot_performance_vs_contamination_few_categories(performance_blind_arr, 
                                                     performance_simple_arr,
                                                     performance_SRR_arr, 
                                                   ratio_labels=np.array([0, 5, 10, 15]), 
                                                   category_labels = np.array(["cable", "wood", "metal_nut"])
                                                   ):
    """Plot performance as a function of the anomaly ratio.
    Args:
        results_arr (numpy array): (runs, contamination ratios, categories)
        ratio_labels (numpy array): 
            Defaults to np.array([0, 5, 10, 15)
        category_labels (numpy array): 
            Defaults to np.array(["Carpet", "Grid", "Leather", "Tile", "Wood", "Bottle", "Cable", "Capsule", "Hazelnut", "Metal nut",  "Pill",  "Screw",  "Toothbrush", "Transistor",  "Zipper"])
    """
    #%matplotlib inline 
    # Plot setttings
    fontsize = 8
    markersize = 4
    n_col = performance_blind_arr.shape[2]

    # Prepare plots
    fig, axs = plt.subplots(1, n_col, figsize=(10, 3), constrained_layout=True)

    for category_idx in np.arange(category_labels.shape[0]):

        # Prepare data per category
        ratios = ratio_labels
        performance_blind_mean = np.mean(performance_blind_arr[:,:,category_idx], axis=0)
        performance_blind_std = np.std(performance_blind_arr[:,:,category_idx], axis=0)
        performance_simple_mean = np.mean(performance_simple_arr[:,:,category_idx], axis=0)
        performance_simple_std = np.std(performance_simple_arr[:,:,category_idx], axis=0)
        performance_SRR_mean = np.mean(performance_SRR_arr[:,:,category_idx], axis=0)
        performance_SRR_std = np.std(performance_SRR_arr[:,:,category_idx], axis=0)
        

        sx = category_idx

        ## Plot data
        #axs[sx].plot(ratios, performance_mean, marker="o", label="PatchCore", color='C0', markersize = markersize, linestyle='dotted')
        #axs[sx].fill_between(ratios, performance_mean-performance_std, performance_mean+performance_std, color='C0', alpha=0.2, linewidth=0.0)
        
        # Plot data
        axs[sx].plot(ratios, performance_blind_mean, marker="o", label="Blind", color='C0', markersize = markersize, linestyle='dotted')
        axs[sx].fill_between(ratios, performance_blind_mean-performance_blind_std, performance_blind_mean+performance_blind_std, color='C0', alpha=0.2, linewidth=0.0)
        axs[sx].plot(ratios, performance_simple_mean, marker="o", label="Simple", color='C1', markersize = markersize, linestyle='dotted')
        axs[sx].fill_between(ratios, performance_simple_mean-performance_simple_std, performance_simple_mean+performance_simple_std, color='C1', alpha=0.2, linewidth=0.0)

        axs[sx].plot(ratios, performance_SRR_mean, marker="o", label="SRR Light", color='C2', markersize = markersize, linestyle='dotted')
        axs[sx].fill_between(ratios, performance_SRR_mean-performance_SRR_std, performance_SRR_mean+performance_SRR_std, color='C2', alpha=0.2, linewidth=0.0)


        # Format plot
        axs[sx].grid()
        axs[sx].set_title(category_labels[category_idx], fontsize=fontsize, fontweight="bold")
        axs[sx].legend(fontsize=fontsize, loc="lower left")
        axs[sx].set_xlim([-0.5, 15.5])
        axs[sx].set_ylim([0.9, 1.01])
        axs[sx].set_xlabel("Anomaly ratio [%]", fontsize=fontsize)
        axs[sx].set_ylabel("AUROC [-]", fontsize=fontsize)
        axs[sx].set_xticks(ratio_labels)
        axs[sx].xaxis.set_tick_params(labelsize=fontsize)
        axs[sx].yaxis.set_tick_params(labelsize=fontsize)



def plot_performance_vs_contamination_each_category(performance_blind_arr, 
                                                    performance_simple_arr,
                                                    performance_SRR_arr,
                                                    ratio_labels=np.array([0, 5, 10, 15]), 
                                                    category_labels = np.array(["Carpet", "Grid", "Leather", "Tile", "Wood", "Bottle", "Cable", "Capsule", "Hazelnut", "Metal nut",  "Pill",  "Screw",  "Toothbrush", "Transistor",  "Zipper"])
                                                    ):
    """Plot performance as a function of the anomaly ratio.
    Args:
        results_arr (numpy array): (runs, contamination ratios, categories)
        ratio_labels (numpy array): 
            Defaults to np.array([0, 5, 10, 15)
        category_labels (numpy array): 
            Defaults to np.array(["Carpet", "Grid", "Leather", "Tile", "Wood", "Bottle", "Cable", "Capsule", "Hazelnut", "Metal nut",  "Pill",  "Screw",  "Toothbrush", "Transistor",  "Zipper"])
    """

    # Plot setttings
    fontsize = 8
    markersize = 4
    n_col = 5

    # Prepare plots
    fig, axs = plt.subplots(int(performance_blind_arr.shape[2]/n_col), n_col, figsize=(8*n_col/3, 12*3/n_col), constrained_layout=True)

    # Prepare data
    ratios = ratio_labels

    
    for category_idx in np.arange(category_labels.shape[0]):

        # Prepare data per category
        ratios = ratio_labels
        performance_blind_mean = np.mean(performance_blind_arr[:,:,category_idx], axis=0)
        performance_blind_std = np.std(performance_blind_arr[:,:,category_idx], axis=0)
        performance_simple_mean = np.mean(performance_simple_arr[:,:,category_idx], axis=0)
        performance_simple_std = np.std(performance_simple_arr[:,:,category_idx], axis=0)
        performance_SRR_mean = np.mean(performance_SRR_arr[:,:,category_idx], axis=0)
        performance_SRR_std = np.std(performance_SRR_arr[:,:,category_idx], axis=0)
        

        #sx = category_idx

        sx = int(np.mod(category_idx, n_col))
        sy = int(np.floor(category_idx/n_col))

        ## Plot data
        #axs[sx].plot(ratios, performance_mean, marker="o", label="PatchCore", color='C0', markersize = markersize, linestyle='dotted')
        #axs[sx].fill_between(ratios, performance_mean-performance_std, performance_mean+performance_std, color='C0', alpha=0.2, linewidth=0.0)
        
        # Plot data
        axs[sy,sx].plot(ratios, performance_blind_mean, marker="o", label="Blind", color='C0', markersize = markersize, linestyle='dotted')
        axs[sy,sx].fill_between(ratios, performance_blind_mean-performance_blind_std, performance_blind_mean+performance_blind_std, color='C0', alpha=0.2, linewidth=0.0)
        axs[sy,sx].plot(ratios, performance_simple_mean, marker="o", label="Simple", color='C1', markersize = markersize, linestyle='dotted')
        axs[sy,sx].fill_between(ratios, performance_simple_mean-performance_simple_std, performance_simple_mean+performance_simple_std, color='C1', alpha=0.2, linewidth=0.0)

        axs[sy,sx].plot(ratios, performance_SRR_mean, marker="o", label="SRR Light", color='C2', markersize = markersize, linestyle='dotted')
        axs[sy,sx].fill_between(ratios, performance_SRR_mean-performance_SRR_std, performance_SRR_mean+performance_SRR_std, color='C2', alpha=0.2, linewidth=0.0)


        # Format plot
        axs[sy,sx].grid()
        axs[sy,sx].set_title(category_labels[category_idx], fontsize=fontsize, fontweight="bold")
        axs[sy,sx].legend(fontsize=fontsize, loc="lower left")
        axs[sy,sx].set_xlim([-0.5, 15.5])
        axs[sy,sx].set_ylim([0.85, 1.01])
        axs[sy,sx].set_xlabel("Anomaly ratio [%]", fontsize=fontsize)
        axs[sy,sx].set_ylabel("AUROC [-]", fontsize=fontsize)
        axs[sy,sx].set_xticks(ratio_labels)
        axs[sy,sx].xaxis.set_tick_params(labelsize=fontsize)
        axs[sy,sx].yaxis.set_tick_params(labelsize=fontsize)











def bkp_plot_performance_vs_contamination_each_category(performance_arr, 
                                                   ratio_labels=np.array([0, 5, 10, 15]), 
                                                   category_labels = np.array(["Carpet", "Grid", "Leather", "Tile", "Wood", "Bottle", "Cable", "Capsule", "Hazelnut", "Metal nut",  "Pill",  "Screw",  "Toothbrush", "Transistor",  "Zipper"])
                                                   ):
    """Plot performance as a function of the anomaly ratio.
    Args:
        results_arr (numpy array): (runs, contamination ratios, categories)
        ratio_labels (numpy array): 
            Defaults to np.array([0, 5, 10, 15)
        category_labels (numpy array): 
            Defaults to np.array(["Carpet", "Grid", "Leather", "Tile", "Wood", "Bottle", "Cable", "Capsule", "Hazelnut", "Metal nut",  "Pill",  "Screw",  "Toothbrush", "Transistor",  "Zipper"])
    """

    # Plot setttings
    fontsize = 8
    markersize = 4
    n_col = 5

    # Prepare plots
    fig, axs = plt.subplots(int(performance_arr.shape[2]/n_col), n_col, figsize=(8*n_col/3, 12*3/n_col), constrained_layout=True)

    # Prepare data
    ratios = ratio_labels

    
    for category_idx in np.arange(category_labels.shape[0]):

        # Prepare data per category
        ratios = ratio_labels
        performance_mean = np.mean(performance_arr[:,:,category_idx], axis=0)
        performance_std = np.std(performance_arr[:,:,category_idx], axis=0)


        sx = int(np.mod(category_idx, n_col))
        sy = int(np.floor(category_idx/n_col))

        # Plot data
        axs[sy,sx].plot(ratios, performance_mean, marker="o", label="PatchCore", color='C0', markersize = markersize, linestyle='dotted')
        axs[sy,sx].fill_between(ratios, performance_mean-performance_std, performance_mean+performance_std, color='C0', alpha=0.2, linewidth=0.0)
        
        # Format plot
        axs[sy,sx].grid()
        axs[sy,sx].set_title(category_labels[category_idx], fontsize=fontsize, fontweight="bold")
        axs[sy,sx].legend(fontsize=fontsize, loc="lower left")
        axs[sy,sx].set_xlim([-0.5, 15.5])
        axs[sy,sx].set_ylim([0.9, 1.01])
        axs[sy,sx].set_xlabel("Anomaly ratio [%]", fontsize=fontsize)
        axs[sy,sx].set_ylabel("AUROC [-]", fontsize=fontsize)
        axs[sy,sx].xaxis.set_tick_params(labelsize=fontsize)
        axs[sy,sx].yaxis.set_tick_params(labelsize=fontsize)

