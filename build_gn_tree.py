"""Builds the NeuralNetwork Geometry Nodes tree and saves it to a .blend.

Run inside Blender 5.1 (Scripting workspace → Open → Run Script) to
regenerate `neural_network/NeuralNetwork.blend`. Edit HIDDEN_COUNT below
if you want the bundled default to differ from 3.

Core tree-building logic lives in `neural_network/tree_builder.py` and is
also used at runtime by the addon's Rebuild Tree operator.
"""

from __future__ import annotations

import importlib
import os
import sys

import bpy

HIDDEN_COUNT = 3

OUTPUT_BLEND_RELPATH = os.path.join("neural_network", "NeuralNetwork.blend")

OUTPUT_DIR_OVERRIDE = ""


def _resolve_output_dir() -> str:
    if OUTPUT_DIR_OVERRIDE:
        return OUTPUT_DIR_OVERRIDE

    for text in bpy.data.texts:
        fp = text.filepath or ""
        if fp and os.path.basename(fp) == "build_gn_tree.py":
            resolved = bpy.path.abspath(fp)
            if os.path.isfile(resolved):
                return os.path.dirname(resolved)

    f = globals().get("__file__", "")
    if f and os.path.isfile(f):
        return os.path.dirname(os.path.abspath(f))

    blend_path = bpy.data.filepath
    if blend_path:
        parent = os.path.dirname(blend_path)
        if os.path.isdir(os.path.join(parent, "neural_network")):
            return parent

    raise RuntimeError(
        "Could not auto-detect the repo root. Set OUTPUT_DIR_OVERRIDE at the "
        "top of build_gn_tree.py to the absolute path of your repo."
    )


def _load_tree_builder(repo_root: str):
    """Import neural_network.tree_builder without requiring the addon to be enabled."""
    addon_pkg = os.path.join(repo_root, "neural_network")
    if not os.path.isdir(addon_pkg):
        raise RuntimeError(f"Missing '{addon_pkg}' folder.")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if "neural_network.tree_builder" in sys.modules:
        return importlib.reload(sys.modules["neural_network.tree_builder"])
    return importlib.import_module("neural_network.tree_builder")


def save_blend() -> str:
    out_dir = _resolve_output_dir()
    tree_builder = _load_tree_builder(out_dir)
    group = tree_builder.build_tree(hidden_count=HIDDEN_COUNT)
    group.use_fake_user = True

    datablocks: set = {group}
    material = bpy.data.materials.get(tree_builder.INPUT_MATERIAL_NAME)
    if material is not None:
        material.use_fake_user = True
        datablocks.add(material)

    out_path = os.path.join(out_dir, OUTPUT_BLEND_RELPATH)
    bpy.data.libraries.write(out_path, datablocks, fake_user=True, compress=True)
    return out_path


if __name__ == "__main__":
    path = save_blend()
    print(f"[build_gn_tree] Saved: {path} (hidden_count={HIDDEN_COUNT})")
