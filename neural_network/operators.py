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
        tree_builder.ensure_input_material()
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


class NN_OT_load_sample(bpy.types.Operator):
    """Load one sample from the training dataset into the input-layer neurons."""

    bl_idname = "nn.load_sample"
    bl_label = "Show Sample"
    bl_options = {"REGISTER", "UNDO"}

    sample_index: bpy.props.IntProperty(name="Sample Index", default=0, min=0)

    @classmethod
    def poll(cls, context):
        return get_nn_modifier(context.active_object) is not None

    def execute(self, context):
        import os

        import numpy as np

        obj = context.active_object
        mod = get_nn_modifier(obj)
        settings = context.scene.nn_training
        path = bpy.path.abspath(settings.dataset_path)
        if not path or not os.path.exists(path):
            self.report({"ERROR"}, "Dataset path not set. Set it in the Training panel.")
            return {"CANCELLED"}

        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".npz":
                data = np.load(path)
                if "X" not in data.files:
                    raise ValueError("npz missing 'X'")
                X = data["X"]
            elif ext == ".csv":
                arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
                X = arr[:, :-1]
            elif ext == ".pt":
                import torch

                loaded = torch.load(path, map_location="cpu")
                if isinstance(loaded, dict):
                    X = loaded["X"].numpy() if hasattr(loaded["X"], "numpy") else loaded["X"]
                else:
                    X = loaded[0].numpy() if hasattr(loaded[0], "numpy") else loaded[0]
            else:
                raise ValueError(f"unsupported dataset extension {ext}")
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to read dataset: {exc}")
            return {"CANCELLED"}

        if self.sample_index >= len(X):
            self.report(
                {"ERROR"},
                f"Sample {self.sample_index} out of range (dataset has {len(X)} samples).",
            )
            return {"CANCELLED"}

        raw_sample = np.asarray(X[self.sample_index]).flatten().astype(np.float32)
        sample = raw_sample.copy()
        input_size = len(sample)

        try:
            grid_ident = socket_identifier(mod.node_group, "Input Grid")
            input_size_ident = socket_identifier(mod.node_group, "Input Size")
            if grid_ident and input_size_ident:
                rows = int(mod[grid_ident])
                tree_input_size = int(mod[input_size_ident])
                cols = (tree_input_size + rows - 1) // rows if rows > 0 else 0
                if rows > 0 and cols > 0 and rows * cols == input_size:
                    img = sample.reshape(rows, cols)
                    img = np.rot90(img, k=-1)
                    sample = img.flatten().astype(np.float32)
        except Exception:
            pass

        mesh = obj.data
        if len(mesh.vertices) < input_size:
            mesh.vertices.add(input_size - len(mesh.vertices))

        attr = mesh.attributes.get(tree_builder.INPUT_ATTRIBUTE)
        if attr is None:
            attr = mesh.attributes.new(
                name=tree_builder.INPUT_ATTRIBUTE, type="FLOAT", domain="POINT"
            )

        values = np.zeros(len(mesh.vertices), dtype=np.float32)
        values[:input_size] = sample
        attr.data.foreach_set("value", values)

        out_msg = ""
        try:
            from . import nn_training, training_operator

            model_path = training_operator.model_path_for_dataset(path)
            if not os.path.exists(model_path):
                self.report({"WARNING"}, f"No model at {model_path}. Train first.")
            elif not nn_training.has_torch():
                self.report({"WARNING"}, "PyTorch not available for prediction.")
            else:
                model, layer_sizes = nn_training.load_model(model_path)
                if not layer_sizes or layer_sizes[0] != input_size:
                    self.report(
                        {"WARNING"},
                        f"Model input size {layer_sizes[0] if layer_sizes else '?'} "
                        f"!= current Input Size {input_size}.",
                    )
                else:
                    probs = nn_training.predict(model, raw_sample[:layer_sizes[0]])
                    out_attr = mesh.attributes.get(tree_builder.OUTPUT_ATTRIBUTE)
                    if out_attr is None:
                        out_attr = mesh.attributes.new(
                            name=tree_builder.OUTPUT_ATTRIBUTE,
                            type="FLOAT",
                            domain="POINT",
                        )
                    out_values = np.zeros(len(mesh.vertices), dtype=np.float32)
                    out_values[: len(probs)] = probs
                    out_attr.data.foreach_set("value", out_values)
                    pred_class = int(np.argmax(probs))
                    out_msg = f", predicted {pred_class} (p={float(probs[pred_class]):.2f})"
        except Exception as exc:
            self.report({"WARNING"}, f"Prediction failed: {exc}")

        mesh.update()
        obj.update_tag(refresh={"DATA"})

        self.report({"INFO"}, f"Loaded sample {self.sample_index}{out_msg}")
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


_classes = (NN_OT_create, NN_OT_load_sample, NN_OT_rebuild_tree)


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
