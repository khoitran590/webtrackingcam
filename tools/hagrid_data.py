"""Read photos from HaGRID v2 (512 px copy on Hugging Face) with HTTP range requests.

The test and train splits come from different people. Both are sorted by class, so a
class's photos sit in a known band of shards, each holding row groups of 100 photos.

HaGRID v2 is published under its own license (a variant of CC BY-SA 4.0):
https://github.com/hukenovs/hagrid. Photos are cached locally and not redistributed.
"""

import json
import time

PARQUET = "https://huggingface.co/datasets/testdummyvt/hagRIDv2_512px/resolve/main/data/{split}/{split}-{shard:05d}-of-{total:05d}.parquet"
NUM_SHARDS = {"test": 301, "train": 1503}
CLASSES = ["call", "dislike", "fist", "four", "grabbing", "grip", "hand_heart", "hand_heart2", "holy", "like",
           "little_finger", "middle_finger", "mute", "no_gesture", "ok", "one", "palm", "peace", "peace_inverted",
           "point", "rock", "stop", "stop_inverted", "take_picture", "three", "three2", "three3", "three_gun",
           "thumb_index", "thumb_index2", "timeout", "two_up", "two_up_inverted", "xsign"]


def _start_shard(cls, split, position):
    """Shard ``position`` (0..1) of the way through the band of shards holding ``cls``."""
    return int((CLASSES.index(cls) + position) * NUM_SHARDS[split] / len(CLASSES))


def _row_group(cls, split, shard, pick_last=False):
    """Starting at ``shard``, find a row group that holds only ``cls``.
    Returns (shard, group, list of JPEG bytes)."""
    import fsspec
    import pyarrow.parquet as pq

    k, total = CLASSES.index(cls), NUM_SHARDS[split]
    shard = min(total - 1, shard)
    fs = fsspec.filesystem("http")
    for _ in range(6):
        with fs.open(PARQUET.format(split=split, shard=shard, total=total), block_size=1 << 20) as f:
            pf = pq.ParquetFile(f)
            labels = pf.read(columns=["label"]).column("label").to_pylist()
            size = pf.metadata.row_group(0).num_rows
            groups = [g for g in range(pf.metadata.num_row_groups) if set(labels[g * size:(g + 1) * size]) == {k}]
            if groups:
                group = groups[-1] if pick_last else groups[0]
                images = pf.read_row_group(group, columns=["image"]).column("image").to_pylist()
                return shard, group, [image["bytes"] for image in images]
            shard += 1 if min(labels) < k else -1
    raise RuntimeError(f"could not find a row group of class {cls!r} in the {split} split")


def fetch(cls, out_dir, shard_offset):
    """Save one 100-photo row group of ``cls`` from the test split into out_dir.
    shard_offset 0 gives the calibration photos; other offsets give held-out shards."""
    done = out_dir / ".complete"
    if done.exists():
        return
    shard, group, images = _row_group(cls, "test", _start_shard(cls, "test", 0.5) + shard_offset,
                                      pick_last=bool(shard_offset))
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        (out_dir / f"s{shard:03d}_g{group}_{i:03d}.jpg").write_bytes(image)
    done.touch()


_STATS = {}  # (split, shard) -> [(min, max) class index of each row group]


def _retry(read, attempts=7):
    """Call ``read``, waiting and retrying when Hugging Face rate-limits (429) or hiccups (5xx)."""
    for attempt in range(attempts):
        try:
            return read()
        except Exception as e:  # aiohttp.ClientResponseError, possibly wrapped by pyarrow
            status = getattr(e, "status", None)
            if status not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            wait = (getattr(e, "headers", None) or {}).get("Retry-After")
            time.sleep(float(wait) if wait and wait.isdigit() else min(120, 10 * 2 ** attempt))


def group_stats(split, shard):
    """(min, max) class index of every row group in one shard. Reads only the parquet
    footer (~1 s), and remembers it; see load_stats / save_stats."""
    import fsspec
    import pyarrow.parquet as pq

    def read():
        url = PARQUET.format(split=split, shard=shard, total=NUM_SHARDS[split])
        with fsspec.filesystem("http").open(url, block_size=1 << 16) as f:
            md = pq.ParquetFile(f).metadata
        col = next(i for i in range(md.num_columns) if md.row_group(0).column(i).path_in_schema == "label")
        return [(md.row_group(g).column(col).statistics.min, md.row_group(g).column(col).statistics.max)
                for g in range(md.num_row_groups)]

    if (split, shard) not in _STATS:
        _STATS[split, shard] = _retry(read)
    return _STATS[split, shard]


def load_stats(path):
    if path.exists():
        _STATS.update({(split, int(shard)): [tuple(g) for g in groups]
                       for key, groups in json.loads(path.read_text()).items() for split, shard in [key.split("/")]})


def save_stats(path):
    path.write_text(json.dumps({f"{split}/{shard}": groups for (split, shard), groups in _STATS.items()}))


def class_shards(cls, split):
    """(first, last) shard holding photos of ``cls``, by binary search over the sorted
    shards: classes differ in size, so their bands of shards are not evenly spaced."""
    k, total = CLASSES.index(cls), NUM_SHARDS[split]
    lo, hi = 0, total - 1
    while lo < hi:  # first shard whose last photo is of class k or later
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if group_stats(split, mid)[-1][1] < k else (lo, mid)
    first, hi = lo, total - 1
    while lo < hi:  # last shard whose first photo is of class k or earlier
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if group_stats(split, mid)[0][0] <= k else (lo, mid - 1)
    return first, lo


def train_row_groups(cls, count, band):
    """``count`` (shard, group) row groups of ``cls`` spread evenly through its ``band`` of
    shards (from class_shards), so they cover many different people. A class with fewer
    shards than ``count`` gets several row groups from the same shard."""
    k, (first, last) = CLASSES.index(cls), band
    n = last - first + 1
    picks = [first + int((i + 0.5) * n / count) for i in range(count)]
    chosen = []
    for shard in sorted(set(picks)):
        pure = [g for g, (lo, hi) in enumerate(group_stats("train", shard)) if lo == hi == k]
        want = picks.count(shard)
        chosen += [(shard, pure[int((j + 0.5) * len(pure) / want)]) for j in range(min(want, len(pure)))]
    return chosen


def read_row_group(split, shard, group):
    """JPEG bytes of every photo in one row group."""
    import fsspec
    import pyarrow.parquet as pq

    def read():
        url = PARQUET.format(split=split, shard=shard, total=NUM_SHARDS[split])
        with fsspec.filesystem("http").open(url, block_size=1 << 20) as f:
            table = pq.ParquetFile(f).read_row_group(group, columns=["image"])
        return [image["bytes"] for image in table.column("image").to_pylist()]

    return _retry(read)
