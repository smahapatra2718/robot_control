#!/usr/bin/env python
"""Bake per-arm 3D model bundles for the web console.

For each arm, load its URDF, convert every VISUAL mesh to glTF (.glb) via trimesh
(meters scale preserved), strip <collision>, and write a self-contained <arm>.urdf
referencing the local .glb files into web/models/<arm>/. STL meshes (no native
material) get the URDF link <color> baked in; DAE/OBJ keep their own materials.

Run once on a box that has the source meshes (the dev machine); the output is
committed so the server serves it everywhere without the robot_descriptions cache.

  uv run scripts/export_web_models.py          # (re)generate
  uv run scripts/export_web_models.py --check  # verify only
"""
import argparse
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robot_common as rc  # noqa: E402

MODELS_DIR = os.path.join(_ROOT, "web", "models")


def _vis_color(vis):
    """RGBA list from a <visual>'s <material><color>, or None."""
    mat = vis.find("material")
    if mat is not None:
        col = mat.find("color")
        if col is not None and col.get("rgba"):
            return [float(x) for x in col.get("rgba").split()]
    return None


def _glb_name(src_abs, seen):
    """Stable, unique .glb basename for a source mesh path."""
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(src_abs))[0])
    name, n = base + ".glb", 1
    used = set(seen.values())
    while name in used and seen.get(src_abs) != name:
        name, n = f"{base}_{n}.glb", n + 1
    return name


def _to_glb(src_abs, out_path, color):
    import numpy as np
    import trimesh
    geom = trimesh.load(src_abs, force="scene")
    if color is not None and src_abs.lower().endswith(".stl"):
        rgba = (np.array(color) * 255).astype(np.uint8)
        for g in geom.geometry.values():
            g.visual = trimesh.visual.ColorVisuals(g, face_colors=rgba)
    geom.export(out_path)


def _convert(tree, resolver, out_dir, out_name):
    """Strip collisions, convert visual meshes to .glb, rewrite refs, write URDF."""
    os.makedirs(out_dir, exist_ok=True)
    root = tree.getroot()
    seen = {}  # src_abs -> glb basename
    for link in root.iter("link"):
        for col in list(link.findall("collision")):
            link.remove(col)
        for vis in link.findall("visual"):
            color = _vis_color(vis)
            for mesh in vis.iter("mesh"):
                fn = mesh.get("filename")
                if not fn:
                    continue
                src = resolver(fn)
                if src.startswith("file://"):
                    src = src[len("file://"):]
                src = os.path.abspath(src)
                if src not in seen:
                    glb = _glb_name(src, seen)
                    _to_glb(src, os.path.join(out_dir, glb), color)
                    seen[src] = glb
                mesh.set("filename", seen[src])
    tree.write(os.path.join(out_dir, out_name), encoding="utf-8", xml_declaration=True)
    return len(seen)


def _ur15_tree():
    """The synthesized UR15 URDF (no local file) as an ElementTree; mesh refs are file://."""
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    u = load_robot_description("ur15_description")
    fd, tmp = tempfile.mkstemp(suffix=".urdf")
    os.close(fd)
    try:
        u.write_xml_file(tmp)
        return ET.parse(tmp)
    finally:
        os.remove(tmp)


def export():
    # UR15 bundle: arm + Hand-E (Hand-E meshes resolve under the project root).
    n = _convert(_ur15_tree(), lambda f: f,
                 os.path.join(MODELS_DIR, "ur15"), "ur15.urdf")
    n += _convert(ET.parse(os.path.join(_ROOT, "urdf", "hande.urdf")),
                  rc.make_mesh_resolver(rc.UR_MESH_DIR_PREFIX),
                  os.path.join(MODELS_DIR, "ur15"), "hande.urdf")
    print(f"ur15: {n} meshes -> web/models/ur15/")

    # GoFa bundle: meshes live under abb_desc/.
    n = _convert(ET.parse(os.path.join(_ROOT, "urdf", "crb15000_5_95.urdf")),
                 rc.make_mesh_resolver(os.path.join(_ROOT, "abb_desc")),
                 os.path.join(MODELS_DIR, "gofa"), "gofa.urdf")
    print(f"gofa: {n} meshes -> web/models/gofa/")


def check():
    """Every committed bundle URDF parses and every referenced .glb exists."""
    bundles = {
        "ur15": ["ur15.urdf", "hande.urdf"],
        "gofa": ["gofa.urdf"],
    }
    total = 0
    for arm, urdfs in bundles.items():
        d = os.path.join(MODELS_DIR, arm)
        for u in urdfs:
            path = os.path.join(d, u)
            assert os.path.isfile(path), f"missing {arm}/{u}"
            root = ET.parse(path).getroot()
            refs = [m.get("filename") for m in root.iter("mesh") if m.get("filename")]
            assert refs, f"{arm}/{u} references no meshes"
            for r in refs:
                assert os.path.isfile(os.path.join(d, r)), f"{arm}/{u} -> missing {r}"
                total += 1
            assert not list(root.iter("collision")), f"{arm}/{u} still has <collision>"
    print(f"OK web/models check — {total} mesh references, all present")


def main():
    ap = argparse.ArgumentParser(description="Bake web-console 3D model bundles.")
    ap.add_argument("--check", action="store_true", help="verify the committed bundles, no writes")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        export()
        check()


if __name__ == "__main__":
    main()
