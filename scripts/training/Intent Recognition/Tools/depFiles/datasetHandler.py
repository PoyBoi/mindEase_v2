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
    # return []

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
    rcParams['figure.figsize'] = 12, 8

    warnings.filterwarnings("ignore")
    # Convert tensors to numpy arrays
    train_accuracies_np = [acc.cpu().numpy() for acc in train_accuracies]
    val_accuracies_np = [acc.cpu().numpy() for acc in val_accuracies]
    train_losses_np = train_losses  # These should already be float values
    val_losses_np = val_losses  # These should already be float values
    train_f1_scores_np = [f1.cpu().numpy() for f1 in train_f1_scores]
    val_f1_scores_np = [f1.cpu().numpy() for f1 in val_f1_scores]

    # Plotting Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_accuracies_np, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies_np, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'accuracy_plot_{epochs}_{avg_val_accuracy:.4f}.png'))
    # plt.close()
    # plt.show()

    # Plotting Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses_np, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses_np, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'loss_plot_{epochs}_{avg_val_accuracy:.4f}.png'))
    # plt.close()
    # plt.show()

    # Plotting F1 Score
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_f1_scores_np, label='Training F1 Score')
    plt.plot(range(1, epochs + 1), val_f1_scores_np, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'f1_score_plot_{epochs}_{avg_val_accuracy:.4f}.png'))
    # plt.close()
    # plt.show()

    print(f"Plots have been saved as 'accuracy_plot_{epochs}_{avg_val_accuracy:.4f}.png', 'loss_plot_{epochs}_{avg_val_accuracy:.4f}.png', and 'f1_score_plot_{epochs}_{avg_val_accuracy:.4f}.png'")
