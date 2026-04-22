import os

import bpy

from . import tree_builder
from .operators import (
    BUNDLED_BLEND,
    NODE_GROUP_NAME,
    get_nn_modifier,
    modifier_input_path,
)


class _NNTabBase:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NN"


class _NNPanelBase(_NNTabBase):
    @classmethod
    def poll(cls, context):
        return get_nn_modifier(context.active_object) is not None


def _current_hidden_count() -> int:
    group = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if group is None:
        return tree_builder.DEFAULT_HIDDEN_COUNT
    return tree_builder.discover_hidden_count(group)


def _layer_names_from_modifier(mod) -> list[str]:
    group = mod.node_group if mod is not None else bpy.data.node_groups.get(NODE_GROUP_NAME)
    if group is None:
        return []
    hidden = tree_builder.discover_hidden_count(group)
    return tree_builder._layer_names(hidden)


class VIEW3D_PT_NN_main(_NNTabBase, bpy.types.Panel):
    bl_label = "Neural Network"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.nn_settings
        mod = get_nn_modifier(context.active_object)
        group = bpy.data.node_groups.get(NODE_GROUP_NAME)

        if mod is not None:
            layout.label(text=f"Active: {context.active_object.name}", icon="CHECKMARK")
            layout.operator("nn.create", text="Add Another", icon="ADD")
        elif group is not None:
            layout.label(text="Node group loaded. Add an object:")
            layout.operator("nn.create", text="Add Neural Network", icon="NODETREE")
        else:
            blend_path = os.path.join(os.path.dirname(__file__), BUNDLED_BLEND)
            if os.path.exists(blend_path):
                layout.label(text="Node group not yet loaded.")
                layout.operator("nn.create", text="Add Neural Network", icon="NODETREE")
            else:
                layout.label(text="No bundled .blend found.", icon="INFO")
                layout.label(text="Use Rebuild Tree below to generate one.")

        box = layout.box()
        box.label(text="Tree Structure", icon="NODETREE")
        row = box.row(align=True)
        row.prop(settings, "hidden_count", text="Hidden Layers")
        op = row.operator("nn.rebuild_tree", text="Rebuild", icon="FILE_REFRESH")
        op.hidden_count = settings.hidden_count

        if group is not None:
            current = tree_builder.discover_hidden_count(group)
            box.label(text=f"Tree currently has {current} hidden layer(s).")


def _draw_mod_prop(layout, mod, socket_name: str, label: str | None = None):
    path = modifier_input_path(mod, socket_name)
    if path is None:
        layout.label(text=f"(missing socket: {socket_name})", icon="ERROR")
        return
    layout.prop(mod, path, text=label if label is not None else socket_name)


class VIEW3D_PT_NN_aspect(_NNPanelBase, bpy.types.Panel):
    bl_label = "Neurons Aspect"

    def draw(self, context):
        mod = get_nn_modifier(context.active_object)
        layout = self.layout

        col = layout.column(align=True)
        col.label(text="Aspect (0=Ico, 1=Sphere, 2=Cube)")
        _draw_mod_prop(col, mod, "Input Aspect", "Input")
        _draw_mod_prop(col, mod, "Hidden Aspect", "Hidden")
        _draw_mod_prop(col, mod, "Output Aspect", "Output")

        col = layout.column(align=True)
        col.label(text="Mesh Size")
        _draw_mod_prop(col, mod, "Input Mesh Size", "Input")
        _draw_mod_prop(col, mod, "Hidden Mesh Size", "Hidden")
        _draw_mod_prop(col, mod, "Output Mesh Size", "Output")

        col = layout.column(align=True)
        col.label(text="Connections")
        _draw_mod_prop(col, mod, "Connection Visibility", "Visible")
        _draw_mod_prop(col, mod, "Connection Radius", "Thickness")
        _draw_mod_prop(col, mod, "Weight Scale", "Weight Scale (0=off)")

        col = layout.column(align=True)
        col.label(text="Spacing")
        _draw_mod_prop(col, mod, "Layer Spacing", "Layer")
        _draw_mod_prop(col, mod, "Neuron Spacing", "Neuron")


class VIEW3D_PT_NN_size(_NNPanelBase, bpy.types.Panel):
    bl_label = "Layer Sizes"

    def draw(self, context):
        mod = get_nn_modifier(context.active_object)
        col = self.layout.column(align=True)
        for name in _layer_names_from_modifier(mod):
            label = name
            if name.startswith("L"):
                label = f"{name} (0 = skip)"
            _draw_mod_prop(col, mod, f"{name} Size", label)


class VIEW3D_PT_NN_grid(_NNPanelBase, bpy.types.Panel):
    bl_label = "Grid"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        mod = get_nn_modifier(context.active_object)
        col = self.layout.column(align=True)
        col.label(text="Grid per layer (1 = line)")
        for name in _layer_names_from_modifier(mod):
            _draw_mod_prop(col, mod, f"{name} Grid", name)


class NNSettings(bpy.types.PropertyGroup):
    hidden_count: bpy.props.IntProperty(
        name="Hidden Layers",
        description="Number of hidden layers to build when clicking Rebuild Tree",
        default=tree_builder.DEFAULT_HIDDEN_COUNT,
        min=0,
        soft_max=8,
        max=tree_builder.MAX_HIDDEN_COUNT,
    )


class NNTrainingSettings(bpy.types.PropertyGroup):
    dataset_path: bpy.props.StringProperty(
        name="Dataset Path",
        description="Path to a .npz / .csv / .pt dataset (see README for format)",
        subtype="FILE_PATH",
    )
    epochs: bpy.props.IntProperty(name="Epochs", default=10, min=1, max=10000)
    minibatch: bpy.props.IntProperty(name="Minibatch", default=32, min=1, max=65536)
    learning_rate: bpy.props.FloatProperty(
        name="Learning Rate", default=1e-3, min=1e-6, max=10.0, precision=5
    )
    last_loss: bpy.props.FloatProperty(name="Last Loss", default=-1.0)
    last_accuracy: bpy.props.FloatProperty(name="Last Accuracy", default=-1.0)
    progress: bpy.props.FloatProperty(name="Progress", default=0.0, min=0.0, max=1.0)
    is_training: bpy.props.BoolProperty(name="Is Training", default=False)
    sample_index: bpy.props.IntProperty(
        name="Sample Index",
        description="Dataset sample to display on the input-layer neurons",
        default=0,
        min=0,
    )


class VIEW3D_PT_NN_training(_NNPanelBase, bpy.types.Panel):
    bl_label = "Training (PyTorch)"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        settings = context.scene.nn_training
        layout = self.layout

        col = layout.column(align=True)
        col.prop(settings, "dataset_path")

        col = layout.column(align=True)
        col.prop(settings, "epochs")
        col.prop(settings, "minibatch")
        col.prop(settings, "learning_rate")

        row = layout.row()
        if settings.is_training:
            row.label(text=f"Training… {settings.progress * 100:.0f}%")
            row.operator("nn.stop_training", text="", icon="CANCEL")
        else:
            row.operator("nn.train", icon="PLAY")

        if settings.last_loss >= 0:
            box = layout.box()
            box.label(text=f"Last loss: {settings.last_loss:.4f}")
            box.label(text=f"Last accuracy: {settings.last_accuracy * 100:.2f}%")

        box = layout.box()
        box.label(text="Show dataset sample")
        row = box.row(align=True)
        row.prop(settings, "sample_index", text="Sample")
        op = row.operator("nn.load_sample", text="Show", icon="IMAGE_DATA")
        op.sample_index = settings.sample_index


_classes = (
    NNSettings,
    NNTrainingSettings,
    VIEW3D_PT_NN_main,
    VIEW3D_PT_NN_aspect,
    VIEW3D_PT_NN_size,
    VIEW3D_PT_NN_grid,
    VIEW3D_PT_NN_training,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.nn_settings = bpy.props.PointerProperty(type=NNSettings)
    bpy.types.Scene.nn_training = bpy.props.PointerProperty(type=NNTrainingSettings)


def unregister():
    del bpy.types.Scene.nn_training
    del bpy.types.Scene.nn_settings
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
