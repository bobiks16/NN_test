import os

def check_img_and_label_pair(img_folder_path: str, label_folder_path: str) -> dict:
    lost_files = {"images": [], "labels": []}
    
    images_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(img_folder_path)]
    files_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(label_folder_path)]

    for image in images_filenames:
        if not image in files_filenames:
            lost_files["images"].append(image)
    
    for f in files_filenames:
        if not f in images_filenames:
            lost_files["labels"].append(f)
    
    return(lost_files)