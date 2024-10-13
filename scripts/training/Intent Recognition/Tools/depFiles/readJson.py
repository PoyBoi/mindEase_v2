import json

def load_config(json_file_path):
    with open(json_file_path, 'r') as file:
        config = json.load(file)
    
    customName = config.get("customName")
    file_loc = config.get("file_loc")
    ds_folder = config.get("ds_folder")
    model_name = config.get("model_name")
    train_file_custom_name = config.get("train_file_custom_name")
    val_file_custom_name = config.get("val_file_custom_name")
    test_file_custom_name = config.get("test_file_custom_name")
    className = config.get("className")
    gpuMode = config.get("gpuMode")
    epochs = config.get("epochs")
    learningRate = config.get("learningRate")
    ifPrompt = config.get("ifPrompt")
    
    return {
        "customName": customName,
        "file_loc": file_loc,
        "ds_folder": ds_folder,
        "model_name": model_name,
        "train_file_custom_name": train_file_custom_name,
        "val_file_custom_name": val_file_custom_name,
        "test_file_custom_name": test_file_custom_name,
        "className": className,
        "gpuMode": gpuMode,
        "epochs": epochs,
        "learningRate": learningRate,
        "ifPrompt": ifPrompt
    }