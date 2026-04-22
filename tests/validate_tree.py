"""Validates the NeuralNetwork node group.

Run via:
    blender --background neural_network/NeuralNetwork.blend \\
            --python tests/validate_tree.py

Exits with nonzero status if any assertion fails. Safe to run in CI.
"""

from __future__ import annotations

import sys

import bpy

NODE_GROUP_NAME = "NeuralNetwork"

FIXED_INPUT_SOCKETS = [
    ("Input Size", "NodeSocketInt"),
    ("Input Grid", "NodeSocketInt"),
    ("Output Size", "NodeSocketInt"),
    ("Output Grid", "NodeSocketInt"),
    ("Input Aspect", "NodeSocketInt"),
    ("Hidden Aspect", "NodeSocketInt"),
    ("Output Aspect", "NodeSocketInt"),
    ("Input Mesh Size", "NodeSocketFloat"),
    ("Hidden Mesh Size", "NodeSocketFloat"),
    ("Output Mesh Size", "NodeSocketFloat"),
    ("Connection Visibility", "NodeSocketBool"),
    ("Connection Radius", "NodeSocketFloat"),
    ("Layer Spacing", "NodeSocketFloat"),
    ("Neuron Spacing", "NodeSocketFloat"),
]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    group = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if group is None:
        fail(f"Node group '{NODE_GROUP_NAME}' not found in .blend")

    if group.bl_idname != "GeometryNodeTree":
        fail(f"Node group type is {group.bl_idname}, expected GeometryNodeTree")

    found_inputs = [
        (item.name, item.socket_type)
        for item in group.interface.items_tree
        if item.item_type == "SOCKET" and item.in_out == "INPUT"
    ]
    found_outputs = [
        (item.name, item.socket_type)
        for item in group.interface.items_tree
        if item.item_type == "SOCKET" and item.in_out == "OUTPUT"
    ]

    if (not found_outputs) or found_outputs[0][1] != "NodeSocketGeometry":
        fail(f"Expected one Geometry output, got {found_outputs}")

    expected = list(FIXED_INPUT_SOCKETS)
    hidden_names = {f[0] for f in found_inputs if f[0].startswith("L") and f[0].endswith(" Size")}
    for hn in hidden_names:
        prefix = hn[:-len(" Size")]
        expected.append((f"{prefix} Size", "NodeSocketInt"))
        expected.append((f"{prefix} Grid", "NodeSocketInt"))

    missing = []
    wrong_type = []
    for name, stype in expected:
        match = [f for f in found_inputs if f[0] == name]
        if not match:
            missing.append(name)
        elif match[0][1] != stype:
            wrong_type.append((name, match[0][1], stype))

    if missing:
        fail(f"Missing input sockets: {missing}")
    if wrong_type:
        fail(f"Wrong socket types: {wrong_type}")

    test_obj = bpy.data.objects.new("TestNN", bpy.data.meshes.new("TestNN"))
    test_obj.data.from_pydata([(0.0, 0.0, 0.0)], [], [])
    bpy.context.scene.collection.objects.link(test_obj)
    mod = test_obj.modifiers.new(name="NN", type="NODES")
    mod.node_group = group

    print(f"OK: '{NODE_GROUP_NAME}' has {len(found_inputs)} inputs and "
          f"{len(found_outputs)} outputs; modifier attaches cleanly.")


if __name__ == "__main__":
    main()
