import os

import bpy

from . import tree_builder

NODE_GROUP_NAME = tree_builder.NODE_GROUP_NAME
BUNDLED_BLEND = "NeuralNetwork.blend"
NN_MARKER_PROP = "is_neural_network"


def _bundled_blend_path() -> str:
    return os.path.join(os.path.dirname(__file__), BUNDLED_BLEND)


def _append_node_group() -> bpy.types.NodeTree | None:
    existing = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if existing is not None:
        return existing

    blend_path = _bundled_blend_path()
    if not os.path.exists(blend_path):
        return None

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        if NODE_GROUP_NAME not in data_from.node_groups:
            return None
        data_to.node_groups = [NODE_GROUP_NAME]

    return bpy.data.node_groups.get(NODE_GROUP_NAME)


def get_nn_modifier(obj: bpy.types.Object | None) -> bpy.types.Modifier | None:
    if obj is None or obj.type != "MESH":
        return None
    for mod in obj.modifiers:
        if mod.type == "NODES" and mod.node_group and mod.node_group.name == NODE_GROUP_NAME:
            return mod
    return None


def socket_identifier(node_group: bpy.types.NodeTree, name: str) -> str | None:
    item = node_group.interface.items_tree.get(name)
    return item.identifier if item is not None else None


def set_modifier_input(mod: bpy.types.Modifier, name: str, value) -> bool:
    ident = socket_identifier(mod.node_group, name)
    if ident is None:
        return False
    mod[ident] = value
    return True


def modifier_input_path(mod: bpy.types.Modifier, name: str) -> str | None:
    ident = socket_identifier(mod.node_group, name)
    return f'["{ident}"]' if ident else None


class NN_OT_create(bpy.types.Operator):
    """Add a new Neural Network visualization to the scene."""

    bl_idname = "nn.create"
    bl_label = "Neural Network"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        node_group = _append_node_group()
        if node_group is None:
            self.report(
                {"ERROR"},
                f"Could not find '{NODE_GROUP_NAME}' node group. "
                f"Run build_gn_tree.py inside Blender to generate '{BUNDLED_BLEND}' "
                "and place it next to this addon.",
            )
            return {"CANCELLED"}

        mesh = bpy.data.meshes.new("NeuralNetwork")
        mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
        obj = bpy.data.objects.new("NeuralNetwork", mesh)
        obj[NN_MARKER_PROP] = True
        context.collection.objects.link(obj)

        mod = obj.modifiers.new(name="NeuralNetwork", type="NODES")
        mod.node_group = node_group

        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {"FINISHED"}


class NN_OT_rebuild_tree(bpy.types.Operator):
    """Rebuild the Neural Network node group with the chosen number of hidden layers."""

    bl_idname = "nn.rebuild_tree"
    bl_label = "Rebuild Tree"
    bl_options = {"REGISTER", "UNDO"}

    hidden_count: bpy.props.IntProperty(
        name="Hidden Layers",
        description="Number of hidden layers in the rebuilt tree (0 = direct input→output)",
        default=tree_builder.DEFAULT_HIDDEN_COUNT,
        min=0,
        soft_max=8,
        max=tree_builder.MAX_HIDDEN_COUNT,
    )

    def execute(self, context):
        existing = bpy.data.node_groups.get(NODE_GROUP_NAME)
        snapshots = tree_builder.snapshot_modifier_values(existing) if existing else {}

        try:
            new_group = tree_builder.build_tree(hidden_count=self.hidden_count)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        tree_builder.restore_modifier_values(new_group, snapshots)
        self.report(
            {"INFO"},
            f"Rebuilt Neural Network with {self.hidden_count} hidden layer(s).",
        )
        return {"FINISHED"}


def _add_menu_entry(self, context):
    self.layout.operator(NN_OT_create.bl_idname, icon="NODETREE")


_classes = (NN_OT_create, NN_OT_rebuild_tree)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_add.append(_add_menu_entry)
    try:
        from . import training_operator
        training_operator.register()
    except Exception as exc:
        print(f"[NN] Training operator not registered: {exc}")


def unregister():
    try:
        from . import training_operator
        training_operator.unregister()
    except Exception:
        pass
    bpy.types.VIEW3D_MT_add.remove(_add_menu_entry)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
