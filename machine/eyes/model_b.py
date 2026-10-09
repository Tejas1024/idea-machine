"""Model B (L15.2): ResNet18 - a picture network that learned on ImageNet photos (torchvision) - taught again on the
pictures he answered. PyTorch, on the PC's processor (no graphics card).

    from machine.eyes import model_b
    imgs = model_b.stack(lab["image"])                 # the candle area of each picture, 224 x 448, as numbers
    p, by_seed = model_b.walk(imgs, y, folds)          # its chance of his yes for every guessed chart

What it sees: only the candle area of the picture he answered (config.EYES_CROP: the heading, the question, the price
and date labels are cut away - they are the same on every picture or carry no candle). How it is taught
(config.EYES_NET, fixed before it saw his answers): its last layers (layer3, layer4) and a new last step that gives one
number - the chance of his yes - learn from his answers for `epochs` rounds; the first layers stay as ImageNet taught
them. Each round the picture moves up to `shift_px` dots - never flipped: time runs left to right and price up.
Taught config.EYES_SEEDS times with different random starts; its answer is the average of those.
"""
from __future__ import annotations

import io
from typing import Iterable

import numpy as np

from machine import config

MEAN = (0.485, 0.456, 0.406)          # ImageNet's colour averages - the network learned with these
STD = (0.229, 0.224, 0.225)
PICTURE_SIZE = (1560, 1191)           # every label picture (width, height) - L15.1
PARTS = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


def pixels(jpeg: bytes, crop: tuple[int, int, int, int] | None = None,
           hw: tuple[int, int] | None = None) -> np.ndarray:
    """One picture's candle area as height x width x 3 numbers 0-255."""
    from PIL import Image
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if crop is None and im.size != PICTURE_SIZE:
        raise ValueError(f"a label picture is {PICTURE_SIZE}, this one is {im.size}")
    h, w = hw or config.EYES_IMAGE_HW
    im = im.crop(crop or config.EYES_CROP).resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def stack(jpegs: Iterable[bytes], **kw) -> np.ndarray:
    return np.stack([pixels(b, **kw) for b in jpegs])


def weights_file() -> str:
    """The ImageNet weights' file name (its first 8 hex = the start of its SHA-256, torchvision's own check)."""
    from torchvision.models import ResNet18_Weights
    return ResNet18_Weights[config.EYES_NET["weights"]].url.rsplit("/", 1)[-1]


def net(pretrained: bool = True, train_from: str | None = None):
    """ResNet18 with one number out. Returns (network, its frozen parts)."""
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    train_from = train_from or config.EYES_NET["train_from"]
    if train_from not in PARTS:
        raise ValueError(f"train_from must be one of {PARTS}")
    w = ResNet18_Weights[config.EYES_NET["weights"]] if pretrained else None
    m = resnet18(weights=w)
    m.fc = torch.nn.Linear(m.fc.in_features, 1)
    frozen = [getattr(m, name) for name in PARTS[:PARTS.index(train_from)]]
    for part in frozen:
        for p in part.parameters():
            p.requires_grad_(False)
    return m, frozen


def fit(imgs: np.ndarray, y: np.ndarray, seed: int = 0, *, pretrained: bool = True, settings: dict | None = None,
        threads: int | None = None):
    """Teach one network on these pictures and answers."""
    import torch
    import torch.nn.functional as F
    s = {**config.EYES_NET, **(settings or {})}
    y = np.asarray(y, dtype=np.float32)
    if len(np.unique(y)) < 2:
        raise ValueError("model B needs both yes and no answers to learn from")
    torch.set_num_threads(int(threads or config.EYES_THREADS))
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    m, frozen = net(pretrained, s["train_from"])
    body = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": float(s["lr_body"])},
                             {"params": list(m.fc.parameters()), "lr": float(s["lr_head"])}],
                            weight_decay=float(s["weight_decay"]))
    yt = torch.from_numpy(y)
    n, batch = len(y), int(s["batch"])
    for _ in range(int(s["epochs"])):
        _learning(m, frozen)
        order = rng.permutation(n)
        for i in range(0, n, batch):
            b = order[i:i + batch]
            if len(b) < 2:                 # a lone last picture: batch norm needs two (it learns next round)
                continue
            x = tensor(imgs[b], int(s["shift_px"]), rng)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(m(x).squeeze(1), yt[b])
            loss.backward()
            opt.step()
    m.eval()
    return m


def predict(m, imgs: np.ndarray, batch: int = 32) -> np.ndarray:
    import torch
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            out.append(torch.sigmoid(m(tensor(imgs[i:i + batch], 0, None)).squeeze(1)).numpy())
    return np.concatenate(out).astype(float) if out else np.zeros(0)


def walk(imgs: np.ndarray, y: np.ndarray, folds, seeds: tuple[int, ...] | None = None, *, pretrained: bool = True,
         settings: dict | None = None, say=None) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """The time split: for each fold and seed, a network learns from the fold's train rows and guesses its test
    rows. Returns the average chance over the seeds (NaN where not guessed) and each seed's own."""
    seeds = tuple(config.EYES_SEEDS if seeds is None else seeds)
    y = np.asarray(y)
    by_seed = {s: np.full(len(y), np.nan) for s in seeds}
    for f in folds:
        for s in seeds:
            m = fit(imgs[f.train], y[f.train], s, pretrained=pretrained, settings=settings)
            by_seed[s][f.test] = predict(m, imgs[f.test])
            if say:
                say(f"model B: {f.year}, start {s} done")
    return np.mean(np.stack([by_seed[s] for s in seeds]), axis=0), by_seed


def tensor(batch: np.ndarray, shift: int, rng) -> "torch.Tensor":
    """Pictures (n x h x w x 3, 0-255) -> the network's input; moved up to `shift` dots (white comes in)."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).float().div_(255.0)
    if shift:
        n, _, h, w = x.shape
        padded = torch.nn.functional.pad(x, (shift, shift, shift, shift), value=1.0)
        dx = rng.integers(0, 2 * shift + 1, size=n)
        dy = rng.integers(0, 2 * shift + 1, size=n)
        x = torch.stack([padded[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(n)])
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _learning(m, frozen) -> None:
    """Learning mode for the parts that learn; the frozen parts keep ImageNet's batch-norm numbers."""
    m.train()
    for part in frozen:
        part.eval()
