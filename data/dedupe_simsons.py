import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


TRAIN_ROOT = Path("simpsons_dataset")
TEST_ROOT = Path("kaggle_simpson_testset")
OUTPUT_ROOT = Path("deduped_simpsons")
TRAIN_THRESHOLD = 0.99
TEST_THRESHOLD = 0.99
CROSS_THRESHOLD = 0.99
BATCH_SIZE = 64

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEST_NAME_RE = re.compile(r"(.+)_\d+$")


def clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def test_class_name(path):
    match = TEST_NAME_RE.fullmatch(path.stem)
    if not match:
        raise ValueError(f"Bad test filename: {path.name}")
    return match.group(1)


def collect_train():
    items = []
    for path in sorted(TRAIN_ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            items.append(
                {
                    "class_name": path.parent.name,
                    "path": path,
                    "name": f"{path.parent.parent.name}__{path.parent.name}__{path.name}",
                }
            )
    return items


def collect_test():
    items = []
    for path in sorted(TEST_ROOT.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            items.append(
                {
                    "class_name": test_class_name(path),
                    "path": path,
                    "name": path.name,
                }
            )
    return items


def load_model(device):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1]).to(device)
    model.eval()
    return model, weights.transforms()


def embed_items(items, model, preprocess, device):
    valid_items = []
    all_embeddings = []

    for start in range(0, len(items), BATCH_SIZE):
        batch_items = items[start : start + BATCH_SIZE]
        images = []
        current_items = []

        for item in batch_items:
            try:
                with Image.open(item["path"]) as image:
                    images.append(preprocess(image.convert("RGB")))
                current_items.append(item)
            except Exception:
                pass

        if not images:
            continue

        batch = torch.stack(images).to(device)
        with torch.inference_mode():
            embeddings = model(batch).flatten(1)
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        all_embeddings.append(embeddings.cpu())
        valid_items.extend(current_items)

    if not all_embeddings:
        return valid_items, torch.empty((0, 2048))

    return valid_items, torch.cat(all_embeddings, dim=0)


def group_by_class(items):
    groups = defaultdict(list)
    for i, item in enumerate(items):
        groups[item["class_name"]].append(i)
    return groups


def find_internal_duplicates(items, embeddings, threshold):
    keep = set()
    pairs = []

    for class_name, indices in group_by_class(items).items():
        class_embeddings = embeddings[indices]
        sims = class_embeddings @ class_embeddings.T
        active = [True] * len(indices)

        for i, global_i in enumerate(indices):
            if not active[i]:
                continue
            keep.add(global_i)
            for j in range(i + 1, len(indices)):
                if not active[j]:
                    continue
                if float(sims[i, j]) >= threshold:
                    active[j] = False
                    pairs.append((class_name, items[global_i], items[indices[j]]))

    return sorted(keep), pairs


def find_cross_duplicates(train_items, train_embeddings, test_items, test_embeddings):
    keep_test = set(range(len(test_items)))
    pairs = []

    train_groups = group_by_class(train_items)
    test_groups = group_by_class(test_items)

    for class_name, test_indices in test_groups.items():
        train_indices = train_groups.get(class_name)
        if not train_indices:
            continue

        sims = test_embeddings[test_indices] @ train_embeddings[train_indices].T
        best_scores, best_matches = sims.max(dim=1)

        for local_i, test_index in enumerate(test_indices):
            if float(best_scores[local_i]) >= CROSS_THRESHOLD:
                keep_test.discard(test_index)
                train_index = train_indices[int(best_matches[local_i])]
                pairs.append((class_name, train_items[train_index], test_items[test_index]))

    return sorted(keep_test), pairs


def unique_path(path):
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def copy_item(item, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = unique_path(dst_dir / item["name"])
    shutil.copy2(item["path"], dst)


def copy_pairs(pairs, root, first_name, second_name):
    counts = defaultdict(int)
    for class_name, first, second in pairs:
        counts[class_name] += 1
        pair_dir = root / class_name / f"pair_{counts[class_name]:04d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(first["path"], pair_dir / f"{first_name}{first['path'].suffix.lower()}")
        shutil.copy2(second["path"], pair_dir / f"{second_name}{second['path'].suffix.lower()}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clean_dir(OUTPUT_ROOT)

    train_items = collect_train()
    test_items = collect_test()

    model, preprocess = load_model(device)
    train_items, train_embeddings = embed_items(train_items, model, preprocess, device)
    test_items, test_embeddings = embed_items(test_items, model, preprocess, device)

    train_keep, train_pairs = find_internal_duplicates(train_items, train_embeddings, TRAIN_THRESHOLD)
    train_unique = [train_items[i] for i in train_keep]
    train_unique_embeddings = train_embeddings[train_keep]

    test_keep, test_pairs = find_internal_duplicates(test_items, test_embeddings, TEST_THRESHOLD)
    test_unique = [test_items[i] for i in test_keep]
    test_unique_embeddings = test_embeddings[test_keep]

    test_keep_after_cross, cross_pairs = find_cross_duplicates(
        train_unique, train_unique_embeddings, test_unique, test_unique_embeddings
    )

    for i in train_keep:
        copy_item(train_items[i], OUTPUT_ROOT / "train" / train_items[i]["class_name"])

    for i in test_keep_after_cross:
        copy_item(test_unique[i], OUTPUT_ROOT / "test" / test_unique[i]["class_name"])

    copy_pairs(train_pairs, OUTPUT_ROOT / "duplicates" / "train", "1", "2")
    copy_pairs(test_pairs, OUTPUT_ROOT / "duplicates" / "test", "1", "2")
    copy_pairs(cross_pairs, OUTPUT_ROOT / "duplicates" / "train_vs_test", "train", "test")

    print("done")


if __name__ == "__main__":
    main()
