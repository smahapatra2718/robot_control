#!/usr/bin/env python
"""embed_images.py - inline photos/screenshots into presentation.html as data URIs.

The handbook is deliberately a single self-contained file: no external assets, so it
works emailed, on a machine with no network, or opened straight off a USB stick. That
means images have to be baked in rather than linked.

Drop files into images/ named after the slot they fill, then run this:

    uv run scripts/embed_images.py            # embed whatever exists, report what is missing
    uv run scripts/embed_images.py --prune    # ...and delete placeholders still unfilled
    uv run scripts/embed_images.py --list     # just show the slots and their status

Slot names come from `data-shot="..."` in presentation.html. images/cell-wide.jpg fills
the slot named cell-wide; .jpg/.jpeg/.png/.webp all work.

Re-running is safe and idempotent: replacing images/cell-wide.jpg and running again
swaps the embedded copy. Everything is downscaled to MAX_WIDTH first, because a phone
photo embeds at ~4 MB of base64 and a handful of those make the file unpleasant to
open. A PNG that is still under PNG_MAX_BYTES after resizing stays PNG so screenshot
text keeps its edges; anything larger becomes JPEG.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECK = os.path.join(_ROOT, "presentation.html")
IMG_DIR = os.path.join(_ROOT, "images")

MAX_WIDTH = 1600          # plenty for a projector; keeps the embedded file sane
JPEG_QUALITY = 82
PNG_MAX_BYTES = 900_000   # above this a "screenshot" is really a photo -> re-encode as JPEG
EXTS = (".jpg", ".jpeg", ".png", ".webp")

# One <figure class="shot" data-shot="X"> ... <figcaption>. Group 2 is the body we swap:
# either the dashed placeholder or a previously embedded <img>.
FIG_RE = re.compile(
    r'(<figure class="shot[^"]*" data-shot="([a-z0-9-]+)">\s*)(.*?)(\s*<figcaption>)',
    re.S,
)


def find_source(slot: str) -> str | None:
    for ext in EXTS:
        p = os.path.join(IMG_DIR, slot + ext)
        if os.path.exists(p):
            return p
    return None


def encode(path: str) -> tuple[str, int]:
    """Return (data-uri, encoded_bytes), downscaling and re-encoding where it helps."""
    raw = open(path, "rb").read()
    is_png = path.lower().endswith(".png")

    try:
        from PIL import Image
    except ImportError:
        # No Pillow: embed as-is rather than failing. Warn if it is genuinely large.
        if len(raw) > 2_000_000:
            print(f"    ! {len(raw)/1e6:.1f} MB embedded unresized (Pillow not installed)")
        mime = "image/png" if is_png else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(raw).decode(), len(raw)

    im = Image.open(io.BytesIO(raw))
    if im.width > MAX_WIDTH:
        h = round(im.height * MAX_WIDTH / im.width)
        im = im.resize((MAX_WIDTH, h), Image.LANCZOS)

    # Keep PNG for crisp screenshot text. Judge by the size *after* downscaling, not the
    # original: a 4K screenshot is huge as a file but usually small once resized, and
    # re-encoding UI text as JPEG makes it fuzzy for no real saving.
    data, mime = None, None
    if is_png:
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        if buf.tell() <= PNG_MAX_BYTES:
            data, mime = buf.getvalue(), "image/png"
    if data is None:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        data, mime = buf.getvalue(), "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode(), len(data)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inline images into presentation.html.")
    ap.add_argument("--prune", action="store_true",
                    help="remove placeholders for slots with no image (do this for the final version)")
    ap.add_argument("--list", action="store_true", help="show slot status and exit")
    args = ap.parse_args()

    if not os.path.exists(DECK):
        print(f"no deck at {DECK}", file=sys.stderr)
        return 1
    html = open(DECK, encoding="utf-8").read()

    slots = [(m.group(2), "<img " in m.group(3)) for m in FIG_RE.finditer(html)]
    if not slots:
        print("no image slots found - has the deck been edited?", file=sys.stderr)
        return 1

    if args.list:
        print(f"{len(slots)} slots in {os.path.basename(DECK)}:\n")
        for name, filled in slots:
            src = find_source(name)
            state = "embedded" if filled else ("ready: " + os.path.basename(src) if src else "empty")
            print(f"  {name:20s} {state}")
        print(f"\ndrop files in {IMG_DIR}/ named <slot>.jpg|.png")
        return 0

    os.makedirs(IMG_DIR, exist_ok=True)
    done: list[str] = []
    missing: list[str] = []

    def repl(m: re.Match) -> str:
        head, name, _body, tail = m.group(1), m.group(2), m.group(3), m.group(4)
        src = find_source(name)
        if src is None:
            missing.append(name)
            return m.group(0)
        uri, nbytes = encode(src)
        done.append(f"{name:20s} {os.path.basename(src)}  ->  {nbytes/1024:.0f} KB")
        return f'{head}<img alt="{name}" src="{uri}">{tail}'

    out = FIG_RE.sub(repl, html)

    if args.prune and missing:
        # Drop the whole <figure> for any slot still unfilled.
        for name in missing:
            out = re.sub(
                r'<figure class="shot[^"]*" data-shot="' + re.escape(name) + r'">.*?</figure>\s*',
                "", out, flags=re.S)
        print(f"pruned {len(missing)} unfilled placeholder(s)")

    open(DECK, "w", encoding="utf-8").write(out)

    if done:
        print(f"embedded {len(done)}:")
        for line in done:
            print("  " + line)
    else:
        print("nothing embedded")
    if missing and not args.prune:
        print(f"\nstill empty ({len(missing)}) - add images/<name>.jpg or run --prune:")
        for name in missing:
            print("  " + name)
    print(f"\ndeck is now {os.path.getsize(DECK)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
