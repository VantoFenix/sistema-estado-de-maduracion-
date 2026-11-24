import argparse
import csv
import hashlib
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice
from pathlib import Path, PurePosixPath
from random import Random
import struct
import time
import zipfile
import zlib

from PIL import Image, ImageOps
from zip_http import RangeReader

KAGGLE_URL = 'https://api.kaggle.com/v1/datasets/download/leftin/fruit-ripeness-unripe-ripe-and-rotten?datasetVersionNumber=2'
FRUIT_FOLDERS = {
    'freshapples': ('manzana', 'madura'),
    'freshbanana': ('platano', 'madura'),
    'freshoranges': ('naranja', 'madura'),
    'rottenapples': ('manzana', 'podrida'),
    'rottenbanana': ('platano', 'podrida'),
    'rottenoranges': ('naranja', 'podrida'),
    'unripe apple': ('manzana', 'verde'),
    'unripe banana': ('platano', 'verde'),
    'unripe orange': ('naranja', 'verde'),
}
VEG_FOLDERS = {'1. Bell Pepper': 'pimiento', '4. Tomato': 'tomate'}
VEG_LABELS = {'Damaged': 'podrida', 'Ripe': 'madura', 'Unripe': 'verde'}
VEG_VAL_DAYS = {'pimiento': {'20220602'}, 'tomate': {'20220508', '20220514', '20220519', '20220604'}}
VEG_TEST_DAYS = {'pimiento': {'20220604', '20220606', '20220609'}, 'tomate': {'20220507', '20220520', '20220521', '20220524'}}
FORMAT = '<IHHHHHIIIHH'
HEADER_SIZE = struct.calcsize(FORMAT)


def sample_kaggle(infos, train_count, val_count, test_count):
    candidates = defaultdict(list)
    for info in infos:
        parts = PurePosixPath(info.filename).parts
        if len(parts) == 7 and parts[4] in {'train', 'test'} and parts[5] in FRUIT_FOLDERS:
            if PurePosixPath(info.filename).suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                candidates[(parts[4], parts[5])].append(info)

    rng = Random(42)
    seen = set()
    samples = []
    for key in sorted(candidates):
        rng.shuffle(candidates[key])
    for original in ('test', 'train'):
        for folder, (species, label) in FRUIT_FOLDERS.items():
            accepted = 0
            limit = test_count if original == 'test' else train_count + val_count
            for info in candidates[original, folder]:
                fingerprint = (info.CRC, info.file_size)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                split = (
                    'test' if original == 'test'
                    else 'val' if accepted < val_count
                    else 'train'
                )
                samples.append((info, 'kaggle', species, label, split, original, ''))
                accepted += 1
                if accepted >= limit:
                    break
    return samples


def sample_vegnet(archive):
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if len(path.parts) != 4 or path.suffix.lower() != '.jpg':
            continue
        _, folder, category, _ = path.parts
        if folder in VEG_FOLDERS and category in VEG_LABELS:
            species = VEG_FOLDERS[folder]
            date = path.stem.split('_')[1]
            split = 'test' if date in VEG_TEST_DAYS[species] else 'val' if date in VEG_VAL_DAYS[species] else 'train'
            yield (info, 'vegnet', species, VEG_LABELS[category], split, '', date)


def group_ranges(records):
    ordered = sorted(records, key=lambda record: record[0].header_offset)
    groups = []
    chunk = []
    start = end = 0
    for record in ordered:
        info = record[0]
        offset = info.header_offset
        new_end = offset + info.compress_size + 1024
        if chunk and (offset - end > 180_000 or new_end - start > 4_000_000):
            groups.append((start, end, chunk))
            chunk = []
        if not chunk:
            start = offset
            end = new_end
        chunk.append(record)
        end = max(new_end, end)
    if chunk:
        groups.append((start, end, chunk))
    return groups


def download_group(group, remote):
    start, end, records = group
    for attempt in range(4):
        try:
            data = remote.read_range(start, end - start)
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)

    for record in records:
        info = record[0]
        offset = info.header_offset - start
        fields = struct.unpack_from(FORMAT, data, offset)
        if fields[0] != 0x04034B50:
            raise ValueError(f'Invalid ZIP header for {info.filename}')
        position = offset + HEADER_SIZE + fields[-2] + fields[-1]
        compressed = data[position:position + info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(compressed, -15)
        elif info.compress_type == zipfile.ZIP_STORED:
            raw = compressed
        else:
            raise ValueError(f'Unexpected ZIP compression: {info.compress_type}')
        if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
            raise ValueError(f'Integrity check failed for {info.filename}')
        yield record, raw


def prepare(target, vegnet, train_count, val_count, test_count):
    target.mkdir(parents=True, exist_ok=True)
    remote = RangeReader(KAGGLE_URL)
    with zipfile.ZipFile(remote) as archive:
        samples = sample_kaggle(archive.infolist(), train_count, val_count, test_count)
    groups = group_ranges(samples)
    print(f'Kaggle: {len(samples)} imágenes en {len(groups)} rangos', flush=True)
    rows = []
    seen = set()
    invalid = []

    def write(record, raw):
        info, source, species, label, split, original, date = record
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            return
        filename = f'{source}-{digest[:24]}.jpg'
        path = target / 'images' / filename
        try:
            with Image.open(BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.resize((224, 224), Image.Resampling.BILINEAR).save(path, quality=92)
        except (OSError, ValueError):
            invalid.append(info.filename)
            return
        seen.add(digest)
        rows.append({
            'file': f'images/{filename}', 'source': source, 'species': species,
            'label': label, 'split': split, 'original_split': original,
            'capture_date': date, 'sha256': digest, 'source_path': info.filename,
        })

    (target / 'images').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = (pool.submit(lambda group: list(download_group(group, remote)), group) for group in groups)
        # Keep only a bounded number of compressed ranges and decoded groups in memory.
        pending = deque(islice(futures, 14))
        group_count = 0
        while pending:
            for record, raw in pending.popleft().result():
                write(record, raw)
            group_count += 1
            pending.extend(islice(futures, 1))
            if group_count % 100 == 0:
                print(f'Kaggle: {len(rows)} imágenes preparadas ({group_count}/{len(groups)} rangos)', flush=True)

    with zipfile.ZipFile(vegnet) as archive:
        records = list(sample_vegnet(archive))
        priority = {'test': 0, 'val': 1, 'train': 2}
        for record in sorted(records, key=lambda item: priority[item[4]]):
            write(record, archive.read(record[0]))

    with (target / 'manifest.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    group_splits = defaultdict(set)
    for row in rows:
        if row['source'] == 'vegnet':
            group_splits[row['species'], row['capture_date']].add(row['split'])
    assert all(len(splits) == 1 for splits in group_splits.values())
    print('Imágenes verificadas:', len(rows), 'ilegibles:', len(invalid), flush=True)
    for key, total in sorted(Counter((r['source'],r['split'],r['label']) for r in rows).items()):
        print(key, total, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--vegnet', type=Path, default=Path('vegnet-v1.zip'))
    parser.add_argument('--train-count', type=int, default=210)
    parser.add_argument('--val-count', type=int, default=50)
    parser.add_argument('--test-count', type=int, default=90)
    args = parser.parse_args()
    prepare(args.output, args.vegnet, args.train_count, args.val_count, args.test_count)
