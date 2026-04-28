"""Smoke-tests the addon's bootstrap operator inside Blender.

Run via:
    blender --background --python tests/test_bootstrap.py

The node group is built on demand by the operator, so no bundled .blend
is required.
"""

from __future__ import annotations

import os
import sys

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    import neural_network
    neural_network.register()

    try:
        result = bpy.ops.nn.create()
        if "FINISHED" not in result:
            fail(f"Create operator returned {result}")

        obj = bpy.context.view_layer.objects.active
        if obj is None or obj.name != "NeuralNetwork":
            fail(f"Active object after create: {obj}")

        mods = [m for m in obj.modifiers if m.type == "NODES"]
        if not mods:
            fail("No Geometry Nodes modifier on created object")
        if mods[0].node_group is None or mods[0].node_group.name != "NeuralNetwork":
            fail(f"Modifier points to wrong node group: {mods[0].node_group}")

        print("OK: bootstrap operator attaches NN modifier correctly.")
    finally:
        neural_network.unregister()


if __name__ == "__main__":
    main()
