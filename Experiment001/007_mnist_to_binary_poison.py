import os
import shutil

def main():
    base_dir = "./data/mnist_images"
    target_dir = "./data/mnist_binary_poison"
    label_list = ['0','1']
    min_len_list = 0
    for label in label_list:
        src_train_dir = os.path.join(base_dir, "train", label)
        list_of_files = os.listdir(src_train_dir)
        if min_len_list == 0 or min_len_list > len(list_of_files):
            min_len_list = len(list_of_files)//200

    for label in label_list:
        src_train_dir = os.path.join(base_dir, "train", label)
        list_of_files = os.listdir(src_train_dir)

        # equal split into 200 clients
        # len_list = len(list_of_files)//200
        new_list_of_files = list_of_files[:min_len_list*200]

        # sort index into 200 clients
        new_list = {}
        for idx,file in enumerate(new_list_of_files):
            new_list[idx%200].append(file) if idx%200 in new_list else new_list.update({idx%200:[file]})

        for dataset in range(200):
            # save honest data
            if dataset < 100:
                target_train_dir = os.path.join(target_dir, "train", f"honest_{dataset}", label)
                os.makedirs(target_train_dir, exist_ok=True)
                for file in new_list[dataset]:
                    src_path = os.path.join(src_train_dir, file)
                    dst_path = os.path.join(target_train_dir, file)
                    shutil.copy(src_path, dst_path)
            # save poisoned data
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