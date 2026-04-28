bl_info = {
    "name": "Artificial Neural Network",
    "author": "Ivo Vacca",
    "version": (2, 3, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > NN",
    "description": "Visualize and train neural networks in Blender using Geometry Nodes and PyTorch.",
    "warning": "Requires PyTorch installed into Blender's bundled Python to use the Training panel.",
    "doc_url": "https://github.com/barckley75/Blender_Neural_Network",
    "category": "Node",
}

from . import operators, panel


def register():
    operators.register()
    panel.register()


def unregister():
    panel.unregister()
    operators.unregister()
