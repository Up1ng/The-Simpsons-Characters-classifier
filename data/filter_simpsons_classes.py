import shutil
from pathlib import Path


TRAIN_ROOT = Path("deduped_simpsons/train")
TEST_ROOT = Path("deduped_simpsons/test")
OUTPUT_ROOT = Path("filtered_simpsons")
MIN_TRAIN_IMAGES = 50

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def class_counts(root):
    counts = {}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        count = 0
        for path in class_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                count += 1
        counts[class_dir.name] = count
    return counts


def copy_split(src_root, dst_root, allowed):
    for class_name in sorted(allowed):
        src_dir = src_root / class_name
        dst_dir = dst_root / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(src_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                shutil.copy2(path, dst_dir / path.name)


def main():
    clean_dir(OUTPUT_ROOT)

    train_counts = class_counts(TRAIN_ROOT)
    test_counts = class_counts(TEST_ROOT)

    allowed = []
    for class_name, count in train_counts.items():
        if count >= MIN_TRAIN_IMAGES and class_name in test_counts:
            allowed.append(class_name)

    copy_split(TRAIN_ROOT, OUTPUT_ROOT / "train", allowed)
    copy_split(TEST_ROOT, OUTPUT_ROOT / "test", allowed)

    print("done")


if __name__ == "__main__":
    main()
