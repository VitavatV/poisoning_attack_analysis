import os
import shutil

def main():
    base_dir = "./data/mnist_images"
    target_dir = "./data/mnist_binary_poison"
    label_list = ['0','1']
    for label in label_list:
        src_train_dir = os.path.join(base_dir, "train", label)
        list_of_files = os.listdir(src_train_dir)
        new_list = {}
        for idx,file in enumerate(list_of_files):
            new_list[idx%200].append(file) if idx%200 in new_list else new_list.update({idx%200:[file]})

        for dataset in range(200):
            if dataset < 100:
                target_train_dir = os.path.join(target_dir, "train", f"honest_{dataset}", label)
                os.makedirs(target_train_dir, exist_ok=True)
                for file in new_list[dataset]:
                    src_path = os.path.join(src_train_dir, file)
                    dst_path = os.path.join(target_train_dir, file)
                    shutil.copy(src_path, dst_path)
            elif dataset >= 100:
                poison_label = '1' if label == '0' else '0'
                target_train_dir = os.path.join(target_dir, "train", f"poison_{dataset-100}", poison_label)
                os.makedirs(target_train_dir, exist_ok=True)
                for file in new_list[dataset]:
                    src_path = os.path.join(src_train_dir, file)
                    dst_path = os.path.join(target_train_dir, file)
                    shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    main()