import os
import shutil


def copy_files_to_directory(file_paths, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file_path in file_paths:
        target_file_path = os.path.join(target_dir, os.path.basename(file_path))
        shutil.copy2(file_path, target_file_path)


def read_train_files(train_txt_path):
    file_paths = []
    with open(train_txt_path, "r") as file:
        for line in file:
            file_path = line.strip()
            file_paths.append(file_path)
    return file_paths


txt_paths = ["./splits/train.txt", "./splits/test.txt", "./splits/val.txt"]
target_dirs = ["ccpd_train", "ccpd_test", "ccpd_val"]

for i in range(3):
    txt_path = txt_paths[i]
    target_dir = target_dirs[i]
    file_paths = read_train_files(txt_path)
    copy_files_to_directory(file_paths, target_dir)

print("Files copied successfully to", target_dir)
