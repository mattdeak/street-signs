import argparse
import random
import os
import tqdm
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("DATA_DIR", type=str, help="The directory of the dataset")
parser.add_argument("TARGET_DIR", type=str, help="The target directory of the split")
parser.add_argument("--train_size", default=0.7, type=float)
parser.add_argument("--validation_size", default=0.15, type=float)

args = parser.parse_args()

if not 0 < args.train_size <= 1:
    raise argparse.ArgumentError("Train size must be a ratio [0-1]")


def make_or_clear_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        for file in os.listdir(dir):
            filepath = os.path.join(dir, file)
            try:
                if os.path.isfile(dir):
                    os.unlink(filepath)
            except Exception as e:
                print(e)


def copy_files(filenames, src_dir, dst_dir):
    for filename in tqdm.tqdm(filenames):
        orig_path = os.path.join(src_dir, filename)
        new_path = os.path.join(dst_dir, filename)
        shutil.copyfile(orig_path, new_path)


def extract_and_copy(files, name, split_num, src_dir, target_dir):
    dir_name = os.path.join(target_dir, name)
    make_or_clear_dir(dir_name)

    print(f"Creating {split_num} {name} files in {dir_name}")
    split_files = set(random.sample(files, k=split_num))
    copy_files(split_files, src_dir, dir_name)
    return files - split_files


assert (
    0 < args.train_size + args.validation_size <= 1
), "Invalid: 0 < train_ratio + validation_ratio <= 1 not satisfied"

data_dir = args.DATA_DIR
target_dir = args.TARGET_DIR

files = set(
    [image_name for image_name in os.listdir(data_dir) if image_name.endswith("png")]
)
train_num = int(len(files) * args.train_size)
if args.train_size + args.validation_size == 1:
    validation_num = len(files) - train_num
else:
    validation_num = int(len(files) * args.validation_size)
    test_num = len(files) - train_num - validation_num

remaining_files = extract_and_copy(files, "train", train_num, data_dir, target_dir)
remaining_files = extract_and_copy(
    remaining_files, "validation", validation_num, data_dir, target_dir
)

if test_num > 0:
    extract_and_copy(remaining_files, "test", test_num, data_dir, target_dir)
