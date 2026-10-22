#!/usr/bin/env python3
"""Extract and decode all textures from DA:I chargen bundle."""
import struct
import os
import sys

import numpy as np
from PIL import Image

from renderer.frostbite import FrostbiteArchive, read_toc
from pathlib import Path

GAME_PATH = Path("/mnt/10TBRaid/games/drop/dragon-age-inquisition")
TEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "dai", "textures")


def make_dds_header(width, height, mip_count, fourcc):
    bpp = 16 if fourcc == b'DXT5' else 8
    pitch = max(1, (width + 3) // 4) * bpp
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000

    hdr = b'DDS '
    hdr += struct.pack('<7I', 124, flags, height, width, pitch, 0, mip_count)
    hdr += b'\x00' * 44
    hdr += struct.pack('<2I', 32, 0x4)
    hdr += fourcc
    hdr += struct.pack('<5I', 0, 0, 0, 0, 0)
    hdr += struct.pack('<4I', 0x1000 | 0x8 | 0x400000, 0, 0, 0)
    hdr += struct.pack('<I', 0)
    return hdr


def bc_mipchain_size(sz, bytes_per_block):
    total = 0
    w, h = sz, sz
    while True:
        bw = max(1, (w + 3) // 4)
        bh = max(1, (h + 3) // 4)
        total += bw * bh * bytes_per_block
        if w == 1 and h == 1:
            break
        w = max(1, w // 2)
        h = max(1, h // 2)
    return total


def count_mips(sz):
    n = 1
    while sz > 1:
        sz //= 2
        n += 1
    return n


def decode_texture(name, chunk_data, out_dirs):
    """Try to decode chunk data as a known texture format. Returns True on success."""
    csz = len(chunk_data)

    # Raw RGBA
    for sz in [16, 32, 64, 128, 256, 512]:
        if sz * sz * 4 == csz:
            px = np.frombuffer(chunk_data, dtype=np.uint8).reshape(sz, sz, 4)
            for d in out_dirs:
                Image.fromarray(px[:, :, :3]).save(os.path.join(d, f'{name}.png'))
            print(f'  {name}: {sz}x{sz} RGBA8')
            return True

    # BC3/DXT5 or BC1/DXT1
    for sz in [64, 128, 256, 512, 1024, 2048, 4096]:
        for fourcc, bpb in [(b'DXT5', 16), (b'DXT1', 8)]:
            expected = bc_mipchain_size(sz, bpb)
            if abs(csz - expected) <= 64:
                mc = count_mips(sz)
                dds = make_dds_header(sz, sz, mc, fourcc)
                dds += chunk_data[:expected]

                # Write temp DDS, decode with Pillow, save as PNG
                tmp_path = os.path.join(out_dirs[0], f'{name}.dds')
                with open(tmp_path, 'wb') as f:
                    f.write(dds)
                try:
                    img = Image.open(tmp_path)
                    # Preserve alpha for DXT5 (BC3) textures
                    mode = 'RGBA' if fourcc == b'DXT5' else 'RGB'
                    for d in out_dirs:
                        img.convert(mode).save(os.path.join(d, f'{name}.png'))
                    fmt_name = 'BC3/DXT5' if fourcc == b'DXT5' else 'BC1/DXT1'
                    print(f'  {name}: {sz}x{sz} {fmt_name}')
                    return True
                except Exception:
                    pass
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                return False

    print(f'  {name}: {csz} bytes - UNKNOWN format')
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract DA:I textures")
    parser.add_argument("--preview-dir", default="/tmp/dai", help="Directory for preview PNGs")
    args = parser.parse_args()

    os.makedirs(TEX_DIR, exist_ok=True)
    os.makedirs(args.preview_dir, exist_ok=True)
    out_dirs = [TEX_DIR, args.preview_dir]

    archive = FrostbiteArchive(GAME_PATH)

    # Build chunk GUID -> SHA1 lookup
    toc = read_toc(GAME_PATH / 'Data' / 'Win32' / 'chunks0.toc')
    chunk_lookup = {}
    for item in toc.get('chunks'):
        g, s = item.get('id'), item.get('sha1')
        if g and s:
            chunk_lookup[g] = s

    bundles = archive.read_superbundle('chargen')
    for b in bundles:
        if 'human_female' not in b.path:
            continue
        for c in b.chunks:
            chunk_lookup[bytes.fromhex(c.name)] = c.sha1

        tex_res = [r for r in b.res
                   if any(x in r.name.lower() for x in
                          ['texture', 'tint', 'hed_', 'skin', 'scalp',
                           'scar', 'age', 'rough', 'blight', 'tattoo',
                           'warpaint', 'proxy'])]
        print(f'Found {len(tex_res)} texture resources')

        decoded_count = 0
        for r in sorted(tex_res, key=lambda x: x.name):
            name = r.name.rstrip('\x00').split('/')[-1]

            data = archive.extract_by_sha1(r.sha1)
            if len(data) < 48:
                continue

            guid = data[0x1c:0x2c]
            sha1 = chunk_lookup.get(guid)
            if not sha1:
                print(f'  {name}: no chunk found')
                continue

            chunk_data = archive.extract_by_sha1(sha1)
            if not chunk_data:
                continue

            if decode_texture(name, chunk_data, out_dirs):
                decoded_count += 1

        print(f'\nDecoded {decoded_count}/{len(tex_res)} textures')
        print(f'Saved to {TEX_DIR} and {args.preview_dir}')
        break


if __name__ == '__main__':
    main()
