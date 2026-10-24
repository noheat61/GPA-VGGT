"""Fetch RealEstate10K test clips: YouTube video -> the frames the poses name.

RealEstate10K ships camera poses plus a YouTube URL and per-frame timestamps; the
pixels have to be re-downloaded. Sequences are grouped by video so a video that
several sequences share is fetched once, and each video is deleted as soon as its
frames are out -- peak disk stays at one video, not the whole split.

Only the frames a sequence actually evaluates are written. `--per_seq 10` picks
them uniformly across the sequence (endpoints included), which is what the
10-frame protocol needs, so a 280-timestamp sequence costs 10 JPEGs.

`--id_map` replaces that choice with a published one. The pi3 evaluation branch
ships `datasets/seq-id-maps/Re10K_relpose_seq-id-map_seed42.json`, a
{sequence: [10 integer ids]} map, and `datasets/preprocess/prepare_re10k.py`
defines what those integers index: `lines[1:]` of the sequence metadata, in
timestamp order, 0-based (it writes them out as `images/{idx:04d}.png`). So an id
is a line number in `<meta>/<seq>.txt` and nothing else has to be guessed --
which sequences, which frames and which poses are all fixed by that pair of
files. With `--id_map` the sequence list comes from the map too, and `--num_seq`
/ `--sampling` / `--seed` are ignored.

  python eval/re10k_fetch.py --meta <RealEstate10K/test> --out <dir> \
                             --num_seq 1000 --per_seq 10 --seed 0
  python eval/re10k_fetch.py --meta <RealEstate10K/test> --out <dir> \
                             --id_map Re10K_relpose_seq-id-map_seed42.json

Writes <out>/<seq_id>/{frames/<timestamp_us>.jpg, cameras.txt} and appends one
status line per sequence to <out>/fetch_log.csv, so a rerun resumes and the
attrition rate (dead videos) stays auditable.
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def read_seq(path):
    """-> (youtube_url, [(timestamp_us, raw_pose_line), ...])"""
    lines = path.read_text().splitlines()
    url = lines[0].strip()
    frames = [(int(l.split()[0]), l) for l in lines[1:] if l.strip()]
    return url, frames


def pick(frames, n, sampling="uniform", stride=1, seed=0):
    """Which n frames of a sequence to keep.

    RealEstate10K clips run ~10 s at 30 fps, so "10 frames" means very different
    baselines depending on how they are drawn, and AVGGT does not publish which
    it used. The options exist to calibrate that against their vanilla numbers:

      uniform  spread across the whole clip (widest baselines)
      window   n consecutive frames from a random start, `stride` apart
      head     the first n frames (narrowest baselines)
    """
    if len(frames) <= n:
        return frames
    if sampling == "uniform":
        step = (len(frames) - 1) / (n - 1)
        return [frames[round(i * step)] for i in range(n)]
    span = (n - 1) * stride + 1
    if sampling == "head":
        start = 0
    elif sampling == "window":
        start = random.Random(seed).randrange(0, max(1, len(frames) - span + 1))
    else:
        raise ValueError(f"unknown sampling {sampling}")
    return [frames[min(start + i * stride, len(frames) - 1)] for i in range(n)]


def download(url, tmp_dir, height, stem="vid", cmd=None, client="default"):
    """Grab the smallest stream at or below `height`. Returns the file or None.

    yt-dlp picks the container, so the output is globbed by stem rather than
    guessed -- the extension is only known after the fact.
    """
    for old in tmp_dir.glob(stem + ".*"):
        old.unlink()
    fmt = f"bestvideo[height<={height}]/best[height<={height}]/best"
    # Client choice decides the resolution ceiling, not just whether the fetch
    # works at all. yt-dlp 2026.07.04 got HTTP 403 on the default client's media
    # URLs from this host, and `player_client=android` was the only one served --
    # but YouTube's SABR-only experiment strips every DASH URL from the android
    # response, leaving format 18 alone, so that combination silently caps the
    # whole dataset at 640x360. yt-dlp >= 2026.08.19 fetches the default client's
    # DASH formats without a 403, which is what makes 720p reachable. Pass the
    # newer yt-dlp through --yt_dlp when the environment's copy is older.
    argv = list(cmd or ["yt-dlp"])
    argv += ["-f", fmt, "--no-warnings", "--quiet", "--no-playlist"]
    if client:
        argv += ["--extractor-args", f"youtube:player_client={client}"]
    argv += ["-o", str(tmp_dir / f"{stem}.%(ext)s"), url]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip().splitlines()[-1][:160] if r.stderr else "unknown"
    hits = [h for h in tmp_dir.glob(stem + ".*") if h.stat().st_size > 0]
    return (hits[0], "") if hits else (None, "no output file")


def extract(video, ts_us, out_dir):
    """One JPEG per timestamp. ffmpeg seeks per frame: accurate, and cheap
    because we ask for ~10 frames rather than decoding the whole clip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    got = 0
    for ts in ts_us:
        dst = out_dir / f"{ts}.jpg"
        if dst.exists():
            got += 1
            continue
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{ts / 1e6:.6f}",
             "-i", str(video), "-frames:v", "1", "-q:v", "2", "-y", str(dst)],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            got += 1
    return got


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path, required=True, help="RealEstate10K/test dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num_seq", type=int, default=1000, help="0 = all")
    p.add_argument("--per_seq", type=int, default=10)
    p.add_argument("--sampling", default="uniform",
                   choices=["uniform", "window", "head"])
    p.add_argument("--stride", type=int, default=1,
                   help="frame gap for --sampling window/head")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--yt_dlp", default="yt-dlp",
                   help="yt-dlp command, split on spaces; e.g. "
                        "\"/path/to/python -m yt_dlp\" to use a newer copy "
                        "installed outside the environment")
    p.add_argument("--player_client", default="default",
                   help="YouTube client yt-dlp asks; 'android' is SABR-capped "
                        "at 360p, 'default' serves DASH up to the video's own "
                        "resolution")
    p.add_argument("--id_map", type=Path, default=None,
                   help="published {seq: [frame ids]} JSON; ids are 0-based line "
                        "numbers into <meta>/<seq>.txt. Overrides the sequence "
                        "list and the sampling")
    args = p.parse_args()

    id_map = json.loads(args.id_map.read_text()) if args.id_map else None
    if id_map is not None:
        have = {f.stem for f in args.meta.glob("*.txt")}
        seqs = sorted(id_map)
        missing = [s for s in seqs if s not in have]
        if missing:
            print(f"{len(missing)} of {len(seqs)} sequences have no metadata file "
                  f"(first: {missing[0]})", flush=True)
        seqs = [s for s in seqs if s in have]
    else:
        seqs = sorted(f.stem for f in args.meta.glob("*.txt"))
        random.Random(args.seed).shuffle(seqs)
        if args.num_seq:
            seqs = seqs[:args.num_seq]

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "fetch_log.csv"
    done = set()
    if log_path.exists():
        with open(log_path) as fh:
            done = {r["seq"] for r in csv.DictReader(fh) if r["status"] == "ok"}

    by_video = defaultdict(list)
    for s in seqs:
        if s in done:
            continue
        url, frames = read_seq(args.meta / f"{s}.txt")
        by_video[url].append((s, frames))

    tmp = args.out / "_video"
    tmp.mkdir(exist_ok=True)
    new = not log_path.exists()
    log = open(log_path, "a", newline="")
    w = csv.DictWriter(log, fieldnames=["seq", "url", "status", "frames"])
    if new:
        w.writeheader()

    ok = dead = short = oob = 0
    for i, (url, members) in enumerate(by_video.items(), 1):
        video, err = download(url, tmp, args.height,
                              cmd=args.yt_dlp.split(),
                              client=args.player_client)
        if video is None:
            for s, _ in members:
                w.writerow({"seq": s, "url": url, "status": "dead", "frames": 0})
                dead += 1
            log.flush()
            print(f"[{i}/{len(by_video)}] DEAD {url} ({err}) -> {len(members)} seq", flush=True)
            continue

        for s, frames in members:
            if id_map is not None:
                ids = id_map[s]
                if max(ids) >= len(frames):
                    # The map indexes metadata lines; if the file is shorter the
                    # two are not the same release and the row must not be faked.
                    w.writerow({"seq": s, "url": url, "status": "idx_oob",
                                "frames": 0})
                    oob += 1
                    continue
                sel = [frames[i] for i in ids]
            else:
                sel = pick(frames, args.per_seq, args.sampling,
                           args.stride, args.seed)
            sdir = args.out / s
            n = extract(video, [t for t, _ in sel], sdir / "frames")
            (sdir / "cameras.txt").write_text(
                url + "\n" + "\n".join(l for _, l in sel) + "\n")
            status = "ok" if n == len(sel) else "short"
            ok += status == "ok"
            short += status == "short"
            w.writerow({"seq": s, "url": url, "status": status, "frames": n})
        log.flush()
        video.unlink(missing_ok=True)
        print(f"[{i}/{len(by_video)}] {url} -> {len(members)} seq "
              f"(ok={ok} short={short} dead={dead})", flush=True)

    log.close()
    shutil.rmtree(tmp, ignore_errors=True)
    total = ok + short + dead + oob
    print(f"RE10K_FETCH_DONE ok={ok} short={short} dead={dead} oob={oob} "
          f"survival={ok / total:.1%}" if total else "RE10K_FETCH_DONE nothing to do",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
