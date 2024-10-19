import os, torch, warnings

import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def ds_split_set(
    file_loc, ds_folder, 
    className:str,
    train_file_custom_name:str, val_file_custom_name:str, test_file_custom_name:str
):

    dataset = pd.read_csv(file_loc)

    train_data, temp_data = train_test_split(dataset, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.4, random_state=42)

    train_data.to_csv(os.path.join(ds_folder, train_file_custom_name), index=False)
    val_data.to_csv(os.path.join(ds_folder, val_file_custom_name), index=False)
    test_data.to_csv(os.path.join(ds_folder, test_file_custom_name), index=False)

    print("Datasets have been split and saved successfully.")

    trainfile = os.path.join(ds_folder, train_file_custom_name)
    validfile = os.path.join(ds_folder, val_file_custom_name)
    testfile = os.path.join(ds_folder, test_file_custom_name)

    traindf = pd.read_csv(trainfile)
    validdf = pd.read_csv(validfile)
    testdf = pd.read_csv(testfile)

    trainfeatures = traindf.copy()
    trainlabels = trainfeatures.pop(className)

    testfeatures = testdf.copy()
    testlabels = testfeatures.pop(className)

    validfeatures = validdf.copy()
    validlabels = validfeatures.pop(className)

    return {
        "trainfeatures" : trainfeatures,
        "trainlabels" : trainlabels,
        "testfeatures" : testfeatures,
        "testlabels" : testlabels,
        "validfeatures" : validfeatures,
        "validlabels" : validlabels
    }

import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

def eval_img(
    save_path,
    epochs, avg_val_accuracy,
    train_accuracies, val_accuracies,
    train_losses, val_losses,
    train_f1_scores, val_f1_scores
):
    sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)
    HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
    rcParams['figure.figsize'] = 20, 6  # Increased width to accommodate three subplots

    warnings.filterwarnings("ignore")
    # Convert tensors to numpy arrays
    train_accuracies_np = [acc.cpu().numpy() for acc in train_accuracies]
    val_accuracies_np = [acc.cpu().numpy() for acc in val_accuracies]
    train_losses_np = train_losses  # These should already be float values
    val_losses_np = val_losses  # These should already be float values
    train_f1_scores_np = [f1.cpu().numpy() for f1 in train_f1_scores]
    val_f1_scores_np = [f1.cpu().numpy() for f1 in val_f1_scores]

    # Create a single figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Training and Validation Metrics over Epochs', fontsize=16)

    # Plotting Accuracy
    ax1.plot(range(1, epochs + 1), train_accuracies_np, label='Training Accuracy')
    ax1.plot(range(1, epochs + 1), val_accuracies_np, label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plotting Loss
    ax2.plot(range(1, epochs + 1), train_losses_np, label='Training Loss')
    ax2.plot(range(1, epochs + 1), val_losses_np, label='Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss')
    ax2.legend()
    ax2.grid(True)

    # Plotting F1 Score
    ax3.plot(range(1, epochs + 1), train_f1_scores_np, label='Training F1 Score')
    ax3.plot(range(1, epochs + 1), val_f1_scores_np, label='Validation F1 Score')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score')
    ax3.legend()
    ax3.grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    plot_filename = f'combined_metrics_plot_{epochs}_{avg_val_accuracy:.4f}.png'
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close()

    print(f"Combined plot has been saved as '{plot_filename}'")