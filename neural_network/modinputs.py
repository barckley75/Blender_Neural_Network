"""Read/write Geometry Nodes modifier inputs on any supported Blender version.

Blender <= 5.1 stores them as ID properties: mod["Socket_2"].
Blender 5.2+ stores them as RNA: mod.properties.inputs.Socket_2.value.
"""

import bpy

# True on Blender 5.2 and later, where modifier inputs moved to mod.properties.
USE_RNA_INPUTS = bpy.app.version >= (5, 2, 0)

_MISSING = object()


def _socket_struct(mod, ident):
    """The 5.2+ input struct for `ident`, or None on older Blender."""
    if not USE_RNA_INPUTS:
        return None
    props = getattr(mod, "properties", None)
    inputs = getattr(props, "inputs", None) if props is not None else None
    if inputs is None:
        return None
    sock = getattr(inputs, ident, None)
    if sock is None or not hasattr(sock, "value"):
        return None
    return sock


def get_input(mod, ident, default=_MISSING):
    sock = _socket_struct(mod, ident)
    if sock is not None:
        return sock.value
    try:
        return mod[ident]
    except (KeyError, TypeError):
        if default is _MISSING:
            raise KeyError(ident)
        return default


def set_input(mod, ident, value) -> None:
    sock = _socket_struct(mod, ident)
    if sock is not None:
        sock.value = value
    else:
        mod[ident] = value
    # Re-evaluate the modifier so the viewport updates.
    if mod.id_data is not None:
        mod.id_data.update_tag()


def draw_input(layout, mod, ident, text) -> None:
    sock = _socket_struct(mod, ident)
    if sock is not None:
        layout.prop(sock, "value", text=text)
    else:
        layout.prop(mod, f'["{ident}"]', text=text)
