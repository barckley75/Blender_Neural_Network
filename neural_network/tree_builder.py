"""Programmatically builds the NeuralNetwork Geometry Nodes tree.

Imported by the addon (so the Rebuild Tree operator can regenerate the tree
with a user-chosen number of hidden layers) and by the repo-root
`build_gn_tree.py` script (which just calls into here then saves a .blend).

API target: Blender 5.1 (NodeTreeInterface API + GeometryNodeIndexSwitch +
FunctionNodeIntegerMath + GeometryNodeAttributeDomainSize).
"""

from __future__ import annotations

import bpy

NODE_GROUP_NAME = "NeuralNetwork"
WEIGHT_ATTRIBUTE = "nn_weight"

DEFAULT_HIDDEN_COUNT = 3
MAX_HIDDEN_COUNT = 32


def _layer_names(hidden_count: int) -> list[str]:
    return ["Input"] + [f"L{i + 1}" for i in range(hidden_count)] + ["Output"]


def _build_interface_spec(hidden_count: int) -> list[tuple[str, str, object, object, object]]:
    sockets: list[tuple[str, str, object, object, object]] = [
        ("Input Size", "NodeSocketInt", 3, 1, 4096),
        ("Input Grid", "NodeSocketInt", 1, 1, 128),
    ]
    for i in range(hidden_count):
        sockets.append((f"L{i + 1} Size", "NodeSocketInt", 4 if i == 0 else 1, 0, 4096))
        sockets.append((f"L{i + 1} Grid", "NodeSocketInt", 1, 1, 128))
    sockets += [
        ("Output Size", "NodeSocketInt", 2, 1, 4096),
        ("Output Grid", "NodeSocketInt", 1, 1, 128),
        ("Input Aspect", "NodeSocketInt", 0, 0, 2),
        ("Hidden Aspect", "NodeSocketInt", 1, 0, 2),
        ("Output Aspect", "NodeSocketInt", 0, 0, 2),
        ("Input Mesh Size", "NodeSocketFloat", 0.3, 0.001, 10.0),
        ("Hidden Mesh Size", "NodeSocketFloat", 0.3, 0.001, 10.0),
        ("Output Mesh Size", "NodeSocketFloat", 0.3, 0.001, 10.0),
        ("Connection Visibility", "NodeSocketBool", True, None, None),
        ("Connection Radius", "NodeSocketFloat", 0.3, 0.0, 100.0),
        ("Weight Scale", "NodeSocketFloat", 0.0, 0.0, 1.0),
        ("Layer Spacing", "NodeSocketFloat", 5.0, 0.5, 1000.0),
        ("Neuron Spacing", "NodeSocketFloat", 1.0, 0.5, 1000.0),
    ]
    return sockets


def _reset_tree(group: bpy.types.NodeTree) -> None:
    group.nodes.clear()
    iface = group.interface
    for item in list(iface.items_tree):
        iface.remove(item)


def _declare_interface(group: bpy.types.NodeTree, hidden_count: int) -> None:
    iface = group.interface
    iface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    iface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for name, socket_type, default, min_val, max_val in _build_interface_spec(hidden_count):
        socket = iface.new_socket(name=name, in_out="INPUT", socket_type=socket_type)
        try:
            socket.default_value = default
        except (TypeError, AttributeError):
            pass
        if min_val is not None and hasattr(socket, "min_value"):
            try:
                socket.min_value = min_val
            except (TypeError, AttributeError):
                pass
        if max_val is not None and hasattr(socket, "max_value"):
            try:
                socket.max_value = max_val
            except (TypeError, AttributeError):
                pass


def _new(group, bl_idname, name=None, location=None):
    node = group.nodes.new(bl_idname)
    if name:
        node.name = name
        node.label = name
    if location:
        node.location = location
    return node


def _link(group, a_out, b_in):
    group.links.new(a_out, b_in)


def _build_layer(
    group: bpy.types.NodeTree,
    group_input: bpy.types.Node,
    layer_index: int,
    size_socket: str,
    grid_socket: str,
    aspect_socket: str,
    mesh_size_socket: str,
    y_offset: float,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    out_in = group_input.outputs

    ceil_div = _new(group, "ShaderNodeMath", f"{size_socket} Ceil Div",
                    (400, y_offset + 200))
    ceil_div.operation = "DIVIDE"
    _link(group, out_in[size_socket], ceil_div.inputs[0])
    _link(group, out_in[grid_socket], ceil_div.inputs[1])

    ceil_node = _new(group, "ShaderNodeMath", f"{size_socket} Ceil",
                     (600, y_offset + 200))
    ceil_node.operation = "CEIL"
    _link(group, ceil_div.outputs[0], ceil_node.inputs[0])

    spacing_minus_one_w = _new(group, "ShaderNodeMath", f"{size_socket} W-1",
                               (400, y_offset + 100))
    spacing_minus_one_w.operation = "SUBTRACT"
    _link(group, out_in[grid_socket], spacing_minus_one_w.inputs[0])
    spacing_minus_one_w.inputs[1].default_value = 1.0

    spacing_minus_one_h = _new(group, "ShaderNodeMath", f"{size_socket} H-1",
                               (600, y_offset + 100))
    spacing_minus_one_h.operation = "SUBTRACT"
    _link(group, ceil_node.outputs[0], spacing_minus_one_h.inputs[0])
    spacing_minus_one_h.inputs[1].default_value = 1.0

    size_x = _new(group, "ShaderNodeMath", f"{size_socket} SizeX",
                  (800, y_offset + 100))
    size_x.operation = "MULTIPLY"
    _link(group, spacing_minus_one_w.outputs[0], size_x.inputs[0])
    _link(group, out_in["Neuron Spacing"], size_x.inputs[1])

    size_y = _new(group, "ShaderNodeMath", f"{size_socket} SizeY",
                  (800, y_offset))
    size_y.operation = "MULTIPLY"
    _link(group, spacing_minus_one_h.outputs[0], size_y.inputs[0])
    _link(group, out_in["Neuron Spacing"], size_y.inputs[1])

    grid = _new(group, "GeometryNodeMeshGrid", f"{size_socket} Grid",
                (1000, y_offset))
    _link(group, size_y.outputs[0], grid.inputs["Size X"])
    _link(group, size_x.outputs[0], grid.inputs["Size Y"])
    _link(group, ceil_node.outputs[0], grid.inputs["Vertices X"])
    _link(group, out_in[grid_socket], grid.inputs["Vertices Y"])

    index_node = _new(group, "GeometryNodeInputIndex", f"{size_socket} Index",
                      (1000, y_offset - 200))

    cmp = _new(group, "FunctionNodeCompare", f"{size_socket} Keep",
               (1200, y_offset - 200))
    cmp.data_type = "INT"
    cmp.operation = "LESS_THAN"
    _link(group, index_node.outputs[0], cmp.inputs[2])
    _link(group, out_in[size_socket], cmp.inputs[3])

    delete_geo = _new(group, "GeometryNodeDeleteGeometry", f"{size_socket} Trim",
                      (1400, y_offset))
    delete_geo.domain = "POINT"
    delete_geo.mode = "ALL"
    _link(group, grid.outputs["Mesh"], delete_geo.inputs["Geometry"])

    invert = _new(group, "FunctionNodeBooleanMath", f"{size_socket} Invert",
                  (1400, y_offset - 200))
    invert.operation = "NOT"
    _link(group, cmp.outputs[0], invert.inputs[0])
    _link(group, invert.outputs[0], delete_geo.inputs["Selection"])

    layer_offset_x = _new(group, "ShaderNodeMath", f"{size_socket} OffsetX",
                          (1400, y_offset - 100))
    layer_offset_x.operation = "MULTIPLY"
    layer_offset_x.inputs[0].default_value = float(layer_index)
    _link(group, out_in["Layer Spacing"], layer_offset_x.inputs[1])

    input_pos = _new(group, "GeometryNodeInputPosition", f"{size_socket} InPos",
                     (1500, y_offset - 300))
    sep_xyz = _new(group, "ShaderNodeSeparateXYZ", f"{size_socket} Sep",
                   (1700, y_offset - 300))
    _link(group, input_pos.outputs["Position"], sep_xyz.inputs[0])

    new_pos = _new(group, "ShaderNodeCombineXYZ", f"{size_socket} NewPos",
                   (1700, y_offset))
    _link(group, layer_offset_x.outputs[0], new_pos.inputs["X"])
    _link(group, sep_xyz.outputs["X"], new_pos.inputs["Y"])
    _link(group, sep_xyz.outputs["Y"], new_pos.inputs["Z"])

    set_pos = _new(group, "GeometryNodeSetPosition", f"{size_socket} SetPos",
                   (1900, y_offset))
    _link(group, delete_geo.outputs["Geometry"], set_pos.inputs["Geometry"])
    _link(group, new_pos.outputs[0], set_pos.inputs["Position"])

    ico = _new(group, "GeometryNodeMeshIcoSphere", f"{size_socket} Ico",
               (1800, y_offset + 400))
    ico.inputs["Radius"].default_value = 1.0
    ico.inputs["Subdivisions"].default_value = 2

    uv_sphere = _new(group, "GeometryNodeMeshUVSphere", f"{size_socket} UV",
                     (1800, y_offset + 300))
    uv_sphere.inputs["Radius"].default_value = 1.0
    uv_sphere.inputs["Segments"].default_value = 16
    uv_sphere.inputs["Rings"].default_value = 8

    cube = _new(group, "GeometryNodeMeshCube", f"{size_socket} Cube",
                (1800, y_offset + 200))
    cube.inputs["Size"].default_value = (1.5, 1.5, 1.5)

    aspect_switch = _new(group, "GeometryNodeIndexSwitch", f"{size_socket} AspectSwitch",
                         (2000, y_offset + 300))
    aspect_switch.data_type = "GEOMETRY"
    while len(aspect_switch.index_switch_items) < 3:
        aspect_switch.index_switch_items.new()
    _link(group, out_in[aspect_socket], aspect_switch.inputs[0])
    _link(group, ico.outputs["Mesh"], aspect_switch.inputs[1])
    _link(group, uv_sphere.outputs["Mesh"], aspect_switch.inputs[2])
    _link(group, cube.outputs["Mesh"], aspect_switch.inputs[3])

    scale_vec = _new(group, "ShaderNodeCombineXYZ", f"{size_socket} Scale",
                     (2000, y_offset))
    _link(group, out_in[mesh_size_socket], scale_vec.inputs["X"])
    _link(group, out_in[mesh_size_socket], scale_vec.inputs["Y"])
    _link(group, out_in[mesh_size_socket], scale_vec.inputs["Z"])

    iop = _new(group, "GeometryNodeInstanceOnPoints", f"{size_socket} Instance",
               (2200, y_offset))
    _link(group, set_pos.outputs["Geometry"], iop.inputs["Points"])
    _link(group, aspect_switch.outputs[0], iop.inputs["Instance"])
    _link(group, scale_vec.outputs[0], iop.inputs["Scale"])

    realize = _new(group, "GeometryNodeRealizeInstances", f"{size_socket} Realize",
                   (2400, y_offset))
    _link(group, iop.outputs["Instances"], realize.inputs["Geometry"])

    return realize.outputs[0], set_pos.outputs["Geometry"]


def _build_connection_pair(
    group: bpy.types.NodeTree,
    group_input: bpy.types.Node,
    a_geom: bpy.types.NodeSocket,
    b_geom: bpy.types.NodeSocket,
    pair_offset_socket: bpy.types.NodeSocket,
    x: float,
    y: float,
) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """Returns (curves_with_radius, pair_size_socket) — caller uses pair_size
    to compute the next pair's offset."""
    a_pts = _new(group, "GeometryNodeMeshToPoints", "A->Pts", (x, y + 100))
    _link(group, a_geom, a_pts.inputs["Mesh"])

    b_pts = _new(group, "GeometryNodeMeshToPoints", "B->Pts", (x, y - 300))
    _link(group, b_geom, b_pts.inputs["Mesh"])

    ds_a = _new(group, "GeometryNodeAttributeDomainSize", "M (A)", (x + 200, y + 100))
    ds_a.component = "POINTCLOUD"
    _link(group, a_pts.outputs["Points"], ds_a.inputs["Geometry"])

    ds_b = _new(group, "GeometryNodeAttributeDomainSize", "N (B)", (x + 200, y - 300))
    ds_b.component = "POINTCLOUD"
    _link(group, b_pts.outputs["Points"], ds_b.inputs["Geometry"])

    pair_size = _new(group, "FunctionNodeIntegerMath", "MxN", (x + 400, y + 250))
    pair_size.operation = "MULTIPLY"
    _link(group, ds_a.outputs["Point Count"], pair_size.inputs[0])
    _link(group, ds_b.outputs["Point Count"], pair_size.inputs[1])

    dup = _new(group, "GeometryNodeDuplicateElements", "Dup A*N", (x + 400, y + 100))
    dup.domain = "POINT"
    _link(group, a_pts.outputs["Points"], dup.inputs["Geometry"])
    _link(group, ds_b.outputs["Point Count"], dup.inputs["Amount"])

    line = _new(group, "GeometryNodeMeshLine", "Line2", (x + 400, y - 100))
    line.mode = "OFFSET"
    line.count_mode = "TOTAL"
    line.inputs["Count"].default_value = 2
    line.inputs["Offset"].default_value = (0.1, 0.0, 0.0)

    iop = _new(group, "GeometryNodeInstanceOnPoints", "InstLines", (x + 600, y))
    _link(group, dup.outputs["Geometry"], iop.inputs["Points"])
    _link(group, line.outputs["Mesh"], iop.inputs["Instance"])

    real = _new(group, "GeometryNodeRealizeInstances", "Real", (x + 800, y))
    _link(group, iop.outputs["Instances"], real.inputs["Geometry"])

    idx = _new(group, "GeometryNodeInputIndex", "Idx", (x + 600, y - 500))

    div2 = _new(group, "FunctionNodeIntegerMath", "idx//2", (x + 800, y - 500))
    div2.operation = "DIVIDE_FLOOR"
    _link(group, idx.outputs[0], div2.inputs[0])
    div2.inputs[1].default_value = 2

    mod_n = _new(group, "FunctionNodeIntegerMath", "bIdx=h%N", (x + 1000, y - 500))
    mod_n.operation = "MODULO"
    _link(group, div2.outputs[0], mod_n.inputs[0])
    _link(group, ds_b.outputs["Point Count"], mod_n.inputs[1])

    pos_b = _new(group, "GeometryNodeInputPosition", "PosB", (x + 800, y - 700))

    sample_b_pos = _new(group, "GeometryNodeSampleIndex", "SampleBPos", (x + 1200, y - 500))
    sample_b_pos.data_type = "FLOAT_VECTOR"
    sample_b_pos.domain = "POINT"
    _link(group, b_pts.outputs["Points"], sample_b_pos.inputs["Geometry"])
    _link(group, pos_b.outputs["Position"], sample_b_pos.inputs["Value"])
    _link(group, mod_n.outputs[0], sample_b_pos.inputs["Index"])

    mod2 = _new(group, "FunctionNodeIntegerMath", "idx%2", (x + 800, y - 900))
    mod2.operation = "MODULO"
    _link(group, idx.outputs[0], mod2.inputs[0])
    mod2.inputs[1].default_value = 2

    cmp = _new(group, "FunctionNodeCompare", "IsOdd", (x + 1000, y - 900))
    cmp.data_type = "INT"
    cmp.operation = "EQUAL"
    _link(group, mod2.outputs[0], cmp.inputs[2])
    cmp.inputs[3].default_value = 1

    set_pos = _new(group, "GeometryNodeSetPosition", "MoveOdd", (x + 1400, y))
    _link(group, real.outputs["Geometry"], set_pos.inputs["Geometry"])
    _link(group, cmp.outputs[0], set_pos.inputs["Selection"])
    _link(group, sample_b_pos.outputs["Value"], set_pos.inputs["Position"])

    global_idx = _new(group, "FunctionNodeIntegerMath", "globalIdx", (x + 1200, y - 1100))
    global_idx.operation = "ADD"
    _link(group, pair_offset_socket, global_idx.inputs[0])
    _link(group, div2.outputs[0], global_idx.inputs[1])

    weight_attr = _new(group, "GeometryNodeInputNamedAttribute", "WAttr",
                       (x + 1200, y - 1300))
    weight_attr.data_type = "FLOAT"
    weight_attr.inputs["Name"].default_value = WEIGHT_ATTRIBUTE

    sample_w = _new(group, "GeometryNodeSampleIndex", "SampleW", (x + 1500, y - 1100))
    sample_w.data_type = "FLOAT"
    sample_w.domain = "POINT"
    _link(group, group_input.outputs["Geometry"], sample_w.inputs["Geometry"])
    _link(group, weight_attr.outputs[0], sample_w.inputs["Value"])
    _link(group, global_idx.outputs[0], sample_w.inputs["Index"])

    abs_w = _new(group, "ShaderNodeMath", "|w|", (x + 1700, y - 1100))
    abs_w.operation = "ABSOLUTE"
    _link(group, sample_w.outputs[0], abs_w.inputs[0])

    sqrt_w = _new(group, "ShaderNodeMath", "sqrt|w|", (x + 1800, y - 1100))
    sqrt_w.operation = "POWER"
    _link(group, abs_w.outputs[0], sqrt_w.inputs[0])
    sqrt_w.inputs[1].default_value = 0.5

    amp_w = _new(group, "ShaderNodeMath", "0.5+1.5x", (x + 1900, y - 1100))
    amp_w.operation = "MULTIPLY_ADD"
    _link(group, sqrt_w.outputs[0], amp_w.inputs[0])
    amp_w.inputs[1].default_value = 1.5
    amp_w.inputs[2].default_value = 0.5

    mix_w = _new(group, "ShaderNodeMix", "Mix", (x + 2000, y - 1100))
    mix_w.data_type = "FLOAT"
    mix_w.clamp_factor = True
    _link(group, group_input.outputs["Weight Scale"], mix_w.inputs[0])
    mix_w.inputs[2].default_value = 1.0
    _link(group, amp_w.outputs[0], mix_w.inputs[3])

    mul_r = _new(group, "ShaderNodeMath", "r*mix", (x + 2100, y - 1100))
    mul_r.operation = "MULTIPLY"
    _link(group, group_input.outputs["Connection Radius"], mul_r.inputs[0])
    _link(group, mix_w.outputs[0], mul_r.inputs[1])

    m2c = _new(group, "GeometryNodeMeshToCurve", "M2C", (x + 1600, y))
    _link(group, set_pos.outputs["Geometry"], m2c.inputs["Mesh"])

    profile = _new(group, "GeometryNodeCurvePrimitiveCircle", "Profile",
                   (x + 2300, y - 400))
    profile.mode = "RADIUS"
    profile.inputs["Resolution"].default_value = 6
    profile.inputs["Radius"].default_value = 0.1

    c2m = _new(group, "GeometryNodeCurveToMesh", "C2M", (x + 2500, y))
    _link(group, m2c.outputs["Curve"], c2m.inputs["Curve"])
    _link(group, profile.outputs["Curve"], c2m.inputs["Profile Curve"])
    _link(group, mul_r.outputs[0], c2m.inputs["Scale"])

    return c2m.outputs["Mesh"], pair_size.outputs[0]


def _build_connections(
    group: bpy.types.NodeTree,
    group_input: bpy.types.Node,
    layer_point_outputs: list[bpy.types.NodeSocket],
    y_offset: float,
) -> bpy.types.NodeSocket:
    curve_outputs: list[bpy.types.NodeSocket] = []

    offset_zero = _new(group, "FunctionNodeInputInt", "offset0",
                       (2600, y_offset + 200))
    offset_zero.integer = 0
    pair_offset_socket: bpy.types.NodeSocket = offset_zero.outputs[0]

    for i in range(len(layer_point_outputs) - 1):
        curve, pair_size = _build_connection_pair(
            group,
            group_input,
            layer_point_outputs[i],
            layer_point_outputs[i + 1],
            pair_offset_socket,
            x=2800,
            y=y_offset + i * 1600,
        )
        curve_outputs.append(curve)

        acc = _new(group, "FunctionNodeIntegerMath", f"offset{i + 1}",
                   (5300, y_offset + i * 1600 + 200))
        acc.operation = "ADD"
        _link(group, pair_offset_socket, acc.inputs[0])
        _link(group, pair_size, acc.inputs[1])
        pair_offset_socket = acc.outputs[0]

    join = _new(group, "GeometryNodeJoinGeometry", "Conn Join",
                (5600, y_offset))
    for c in curve_outputs:
        _link(group, c, join.inputs[0])

    return join.outputs[0]


def discover_hidden_count(group: bpy.types.NodeTree) -> int:
    """Inspect a node group and return how many L# layers its interface exposes."""
    count = 0
    i = 1
    while True:
        name = f"L{i} Size"
        if any(item.item_type == "SOCKET" and item.in_out == "INPUT" and item.name == name
               for item in group.interface.items_tree):
            count += 1
            i += 1
        else:
            return count


def build_tree(hidden_count: int = DEFAULT_HIDDEN_COUNT) -> bpy.types.NodeTree:
    """Create or reset the NeuralNetwork node group with `hidden_count` hidden layers.

    Layers: Input, L1..L{hidden_count}, Output. hidden_count may be 0 (direct Input→Output).
    Returns the node group. Idempotent: safe to call repeatedly.
    """
    if hidden_count < 0:
        raise ValueError("hidden_count must be >= 0")
    if hidden_count > MAX_HIDDEN_COUNT:
        raise ValueError(f"hidden_count must be <= {MAX_HIDDEN_COUNT}")

    group = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if group is None:
        group = bpy.data.node_groups.new(NODE_GROUP_NAME, "GeometryNodeTree")
    _reset_tree(group)
    _declare_interface(group, hidden_count)

    group_input = _new(group, "NodeGroupInput", "Group Input", (0, 0))
    group_output = _new(group, "NodeGroupOutput", "Group Output", (6400, 0))

    layer_geo_outputs: list[bpy.types.NodeSocket] = []
    layer_point_outputs: list[bpy.types.NodeSocket] = []

    names = _layer_names(hidden_count)
    for i, name in enumerate(names):
        size_socket = f"{name} Size"
        grid_socket = f"{name} Grid"
        if name == "Input":
            aspect_socket, mesh_size_socket = "Input Aspect", "Input Mesh Size"
        elif name == "Output":
            aspect_socket, mesh_size_socket = "Output Aspect", "Output Mesh Size"
        else:
            aspect_socket, mesh_size_socket = "Hidden Aspect", "Hidden Mesh Size"

        geo_out, pts_out = _build_layer(
            group,
            group_input,
            layer_index=i,
            size_socket=size_socket,
            grid_socket=grid_socket,
            aspect_socket=aspect_socket,
            mesh_size_socket=mesh_size_socket,
            y_offset=-i * 1000,
        )
        layer_geo_outputs.append(geo_out)
        layer_point_outputs.append(pts_out)

    join_all = _new(group, "GeometryNodeJoinGeometry", "Join Layers",
                    (2800, -2000))
    for geo in layer_geo_outputs:
        _link(group, geo, join_all.inputs[0])

    if len(layer_point_outputs) >= 2:
        conn_geo = _build_connections(
            group, group_input, layer_point_outputs, y_offset=-5500
        )
        visibility_switch = _new(group, "GeometryNodeSwitch", "Conn Visibility Switch",
                                 (5900, -2500))
        visibility_switch.input_type = "GEOMETRY"
        _link(group, group_input.outputs["Connection Visibility"],
              visibility_switch.inputs[0])
        _link(group, conn_geo, visibility_switch.inputs[2])

        final_join = _new(group, "GeometryNodeJoinGeometry", "Final Join",
                          (6200, -2000))
        _link(group, join_all.outputs[0], final_join.inputs[0])
        _link(group, visibility_switch.outputs[0], final_join.inputs[0])
        _link(group, final_join.outputs[0], group_output.inputs[0])
    else:
        _link(group, join_all.outputs[0], group_output.inputs[0])

    return group


def snapshot_modifier_values(group: bpy.types.NodeTree) -> dict:
    """Record current modifier socket values by socket NAME across every object
    that uses this group, so we can restore them after a rebuild."""
    id_to_name = {
        item.identifier: item.name
        for item in group.interface.items_tree
        if item.item_type == "SOCKET" and item.in_out == "INPUT"
    }
    snapshots: dict = {}
    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if mod.type == "NODES" and mod.node_group is group:
                snap = {}
                for ident, name in id_to_name.items():
                    try:
                        snap[name] = mod[ident]
                    except (KeyError, TypeError):
                        pass
                snapshots[(obj.name, mod.name)] = snap
    return snapshots


def restore_modifier_values(group: bpy.types.NodeTree, snapshots: dict) -> None:
    name_to_id = {
        item.name: item.identifier
        for item in group.interface.items_tree
        if item.item_type == "SOCKET" and item.in_out == "INPUT"
    }
    for (obj_name, mod_name), snap in snapshots.items():
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        mod = obj.modifiers.get(mod_name)
        if mod is None:
            continue
        mod.node_group = group
        for socket_name, value in snap.items():
            ident = name_to_id.get(socket_name)
            if ident is None:
                continue
            try:
                mod[ident] = value
            except (TypeError, ValueError):
                pass
