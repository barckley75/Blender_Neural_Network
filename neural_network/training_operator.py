import bpy

from . import tree_builder
from .operators import get_nn_modifier, socket_identifier


def _layer_sizes_from_modifier(mod) -> list[int]:
    group = mod.node_group
    hidden = tree_builder.discover_hidden_count(group)
    names = ["Input"] + [f"L{i + 1}" for i in range(hidden)] + ["Output"]
    sizes: list[int] = []
    for name in names:
        ident = socket_identifier(group, f"{name} Size")
        if ident is None:
            continue
        sizes.append(int(mod[ident]))
    return sizes


def _push_weights_to_object(obj, trainer) -> tuple[int, float]:
    """Replace obj's mesh with one carrying an 'nn_weight' per-point attribute.
    Returns (weight_count, max_abs) for UI reporting."""
    import numpy as np

    from . import nn_training

    signed_norm, max_abs = trainer.flat_weights()
    n = int(signed_norm.size)
    if n == 0:
        return 0, 0.0

    preserved_input: "np.ndarray | None" = None
    old_mesh = obj.data
    if old_mesh is not None:
        old_attr = old_mesh.attributes.get(tree_builder.INPUT_ATTRIBUTE)
        if old_attr is not None and len(old_attr.data) > 0:
            preserved_input = np.zeros(len(old_attr.data), dtype=np.float32)
            old_attr.data.foreach_get("value", preserved_input)

    mesh = bpy.data.meshes.new(f"{obj.name}_weights")
    mesh.vertices.add(n)
    attr = mesh.attributes.new(
        name=tree_builder.WEIGHT_ATTRIBUTE, type="FLOAT", domain="POINT"
    )
    attr.data.foreach_set("value", signed_norm)

    if preserved_input is not None:
        in_attr = mesh.attributes.new(
            name=tree_builder.INPUT_ATTRIBUTE, type="FLOAT", domain="POINT"
        )
        values = np.zeros(n, dtype=np.float32)
        copy_n = min(n, preserved_input.size)
        values[:copy_n] = preserved_input[:copy_n]
        in_attr.data.foreach_set("value", values)

    mesh.update()

    obj.data = mesh
    if old_mesh is not None and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

    mod = get_nn_modifier(obj)
    if mod is not None:
        ident = socket_identifier(mod.node_group, "Weight Scale")
        if ident is not None:
            mod[ident] = 1.0

    return n, max_abs


class NN_OT_train(bpy.types.Operator):
    """Train the neural network on the dataset at the given path."""

    bl_idname = "nn.train"
    bl_label = "Start Training"
    bl_options = {"REGISTER"}

    _timer = None
    _trainer = None
    _settings = None
    _obj_name: str = ""

    @classmethod
    def poll(cls, context):
        return get_nn_modifier(context.active_object) is not None

    def execute(self, context):
        try:
            from . import nn_training
        except ImportError as exc:
            self.report({"ERROR"}, f"Could not import training backend: {exc}")
            return {"CANCELLED"}

        if not nn_training.has_torch():
            self.report(
                {"ERROR"},
                "PyTorch is not installed in Blender's Python. "
                "See README for installation instructions.",
            )
            return {"CANCELLED"}

        mod = get_nn_modifier(context.active_object)
        if mod is None:
            self.report({"ERROR"}, "No Neural Network object active.")
            return {"CANCELLED"}

        settings = context.scene.nn_training
        sizes = _layer_sizes_from_modifier(mod)
        sizes = [s for s in sizes if s > 0]
        if len(sizes) < 2:
            self.report({"ERROR"}, "Need at least an input and output layer with size > 0.")
            return {"CANCELLED"}

        try:
            trainer = nn_training.EpochTrainer(
                dataset_path=bpy.path.abspath(settings.dataset_path),
                layer_sizes=sizes,
                epochs=settings.epochs,
                minibatch=settings.minibatch,
                lr=settings.learning_rate,
            )
        except (FileNotFoundError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self._trainer = trainer
        self._settings = settings
        self._obj_name = context.active_object.name
        settings.is_training = True
        settings.progress = 0.0

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC" or not self._settings.is_training:
            return self._cleanup(context, cancelled=True)

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        try:
            result = self._trainer.step()
        except Exception as exc:
            self.report({"ERROR"}, f"Training failed: {exc}")
            return self._cleanup(context, cancelled=True)

        if result is None:
            self._settings.last_loss = self._trainer.last_loss
            self._settings.last_accuracy = self._trainer.last_accuracy
            obj = bpy.data.objects.get(self._obj_name)
            if obj is not None:
                try:
                    n, max_abs = _push_weights_to_object(obj, self._trainer)
                    if n > 0:
                        self.report(
                            {"INFO"},
                            f"Training done. Pushed {n} weights (max |w|={max_abs:.3f}).",
                        )
                except Exception as exc:
                    print(f"[NN] weight push failed: {exc}")
            return self._cleanup(context, cancelled=False)

        epoch, loss, acc = result
        self._settings.last_loss = loss
        self._settings.last_accuracy = acc
        self._settings.progress = epoch / self._trainer.epochs
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _cleanup(self, context, cancelled: bool):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        self._settings.is_training = False
        self._settings.progress = 1.0 if not cancelled else self._settings.progress
        return {"CANCELLED" if cancelled else "FINISHED"}


class NN_OT_stop_training(bpy.types.Operator):
    """Stop an in-progress training run."""

    bl_idname = "nn.stop_training"
    bl_label = "Stop Training"

    def execute(self, context):
        context.scene.nn_training.is_training = False
        return {"FINISHED"}


_classes = (NN_OT_train, NN_OT_stop_training)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
