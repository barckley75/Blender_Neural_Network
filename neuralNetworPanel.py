from bpy import data
import bpy
bl_info = {
    "name": "Artificial Neural Network",
    "author": "Ivo Vacca",
    "version": (0, 1),
    "blender": (2, 93, 1),
    "location": "View3D > NN > controls",
    "description": "Create a Neural Network and Training it.",
    "warning": "This file works library that are still in beta.",
    "doc_url": "https://github.com/barckley75/Blender_Neural_Network",
    "category": "Artificial Neural Network",
}

# --------------------------------------------------------------------------------------------
# ------------------------------------ MODEL OPERATOR ----------------------------------------
# --------------------------------------------------------------------------------------------
# this class creates the operator in bpy.ops
# In bpy.ops will be added in this case node.neurons from bl_name="node.neurons
# The operator in this case will be bpy.ops.node.neurons
# that you can call with f3 operator


class NEURONS_OT_model(bpy.types.Operator):
    """Before train the model be sure that your dataset is well formatted for the network."""
    bl_idname = "node.neurons"
    bl_label = "Neurons Grid"

    # this line permits to Blender to appear the UNDO/REDO panel
    bl_options = {'REGISTER', 'UNDO'}

    # ---------------------------------------------------------------
    # ----------------- NUMBER OF NEURONS ---------------------------
    # ---------------------------------------------------------------

    # input number of neurons
    inputSize: bpy.props.IntProperty(
        name="Dataset Size",
        description="Dataset Size",
        default=3,
        min=1,
        soft_max=5
    )

    # ------- IN----------

    # hidden layer 1 number of neurons
    numberOfNeurons_1: bpy.props.IntProperty(
        name="L1 Size",
        description="Neurons",
        default=1,
        min=1,
        soft_max=5
    )

    # --- HIDDEN LAYERS ----

    # hidden layer 2 number of neurons
    numberOfNeurons_2: bpy.props.IntProperty(
        name="L2 Size",
        description="Neurons",
        default=1,
        min=1,
        soft_max=5
    )

    # hidden layer 3 number of neurons
    numberOfNeurons_3: bpy.props.IntProperty(
        name="L3 Size",
        description="Neurons",
        default=1,
        min=1,
        soft_max=5
    )

    # ------ OUT -----------

    # output number of neurons
    output_x: bpy.props.IntProperty(
        name="Output Size",
        description="Width of the output",
        default=2,
        min=1,
        soft_max=5
    )

    # ---------------------------------------------------
    # ---------------- SHAPE ----------------------------
    # ---------------------------------------------------

    # ------- IN----------
    # input shape
    inputShape: bpy.props.IntProperty(
        name="Dataset Shape",
        description="Shape The Dataset",
        default=1,
        min=1,
        soft_max=5
    )

    # --- HIDDEN LAYERS ----

    # hidden layer 1 shape
    shape_1: bpy.props.IntProperty(
        name="L1 Shape",
        description="Shape",
        default=1,
        min=1,
        soft_max=5
    )

    # hidden layer 2 shape
    shape_2: bpy.props.IntProperty(
        name="L2 Shape",
        description="Shape",
        default=1,
        min=1,
        soft_max=5
    )

    # hidden layer 3 shape
    shape_3: bpy.props.IntProperty(
        name="L3 Shape",
        description="Shape",
        default=1,
        min=1,
        soft_max=5
    )

    # ------ OUT -----------

    # output shape
    output_z: bpy.props.IntProperty(
        name="Output Shape",
        description="Heigth of the output",
        default=1,
        min=1,
        soft_max=5
    )

    # --------------------------------------------

    # this is where you can put all the operations you want to execute
    def execute(self, context):

        # --------------------------------------------------------------------------------------------
        # ----------------------------------------- INPUTS -------------------------------------------
        # --------------------------------------------------------------------------------------------
        inputDataset = bpy.data.node_groups["Artificial Neural Network"]

        # input numbers of neurons
        inputDataset.nodes["Data Input.009"].inputs[0].value = self.inputSize
        # shape
        inputDataset.nodes["Data Input.008"].inputs[0].value = self.inputShape

        # --------------------------------------------------------------------------------------------
        # ----------------------------------------- DEEP LEARNING ------------------------------------
        # --------------------------------------------------------------------------------------------

        # ----------------------------------------- HIDDEN LAYER 1 -----------------------------------
        L1_Neurons = bpy.data.node_groups["Artificial Neural Network"]

        # L3 numbers of neurons
        L1_Neurons.nodes["Data Input.015"].inputs[0].value = self.numberOfNeurons_1
        # shape
        L1_Neurons.nodes["Data Input.016"].inputs[0].value = self.shape_1

        # --------------------------------------------------------------------------------------------
        # ----------------------------------------- OUTPUTS ------------------------------------------
        # --------------------------------------------------------------------------------------------
        nodeOutput = bpy.data.node_groups["Artificial Neural Network"]

        # output numbers of neurons
        nodeOutput.nodes["Data Input.021"].inputs[0].value = self.output_x
        # shape
        nodeOutput.nodes["Data Input.022"].inputs[0].value = self.output_z

        return {'FINISHED'}

# VIEW3D_PT_NN_model_training where the meaning of VIEW3D -> 3D view, P -> panel, T -> type and then the - name of the class


class VIEW3D_PT_NN_model_aspect(bpy.types.Panel):
    """Creates a Panel in the 3D view scene"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'NN'
    bl_label = "Neurons Aspect"

    def draw(self, context):
        AN_node = bpy.data.node_groups["Artificial Neural Network"]

        # --------------------------------------------------------------------------------------------
        # ----------------------------------------- ASPECT ------------------------------------------
        # --------------------------------------------------------------------------------------------

        # Dataset Aspect
        col = self.layout.column(align=True)
        col.label(text='Neurons Aspect Mesh')
        col.prop(AN_node.nodes["Get List Element.001"].inputs[1], 'value',
                 text='Dataset')

        # Hidden Layer Aspect
        col.prop(AN_node.nodes["Get List Element.002"].inputs[1], 'value',
                 text='Hidden Layers')

        # Hidden Layer Aspect
        col.prop(AN_node.nodes["Get List Element.003"].inputs[1], 'value',
                 text='Output')

        # --------------------------------------------------------------------------------------------
        # ----------------------------------------- SIZE OF THE MESH ---------------------------------
        # --------------------------------------------------------------------------------------------

        # Dataset Size Mesh
        col = self.layout.column(align=True)
        col.label(text='Neurons Mesh Size')
        col.prop(AN_node.nodes["Data Input.011"].inputs[0], 'value',
                 text='Dataset')

        # Hidden Layer Size Mesh
        col.prop(AN_node.nodes["Data Input.001"].inputs[0], 'value',
                 text='Hidden Layers')

        # Hidden Layer Size Mesh
        col.prop(AN_node.nodes["Data Input"].inputs[0], 'value',
                 text='Output')

        # --------------------------------------------------------------------------------------------
        # ----------------------------------------- CONNECTIONS ---------------------------------
        # --------------------------------------------------------------------------------------------

        # Hide Connections
        col = self.layout.column(align=True)
        col.label(text='Connections Visibility')
        col.prop(AN_node.nodes["Create List.009"].inputs[0], 'value',
                 text='Visibility')

        # Hide Connections
        col = self.layout.column(align=True)
        col.label(text='Connections Radius')
        col.prop(AN_node.nodes["Float Math"].inputs[0], 'value',
                 text='Radius')


# VIEW3D_PT_NN_model_training where the meaning of VIEW3D -> 3D view, P -> panel, T -> type and then the - name of the class
class VIEW3D_PT_NN_model_size(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'NN'
    bl_label = "Neurons Model Size"

    def draw(self, context):
        AN_node = bpy.data.node_groups["Artificial Neural Network"]

        # Size Dataset
        col = self.layout.column(align=True)
        col.label(text='Size - Dataset')
        col.prop(AN_node.nodes["Data Input.009"].inputs[0], 'value',
                 text='Dataset')

        # Size - Hidden Layer
        col = self.layout.column(align=True)
        col.label(text='Size - Hidden Layer')
        col.prop(AN_node.nodes["Data Input.015"].inputs[0], 'value',
                 text='Hidden Layer')

        # Size - Output
        col = self.layout.column(align=True)
        col.label(text='Size - Output')
        col.prop(AN_node.nodes["Data Input.021"].inputs[0], 'value',
                 text='Output')

# VIEW3D_PT_NN_model_training where the meaning of VIEW3D -> 3D view, P -> panel, T -> type and then the - name of the class


class VIEW3D_PT_NN_model_shape(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'NN'
    bl_label = "Neurons Model Shape"

    def draw(self, context):

        AN_node = bpy.data.node_groups["Artificial Neural Network"]

        # Dataset Shape
        col = self.layout.column(align=True)
        col.label(text='Dataset Shape')
        col.prop(AN_node.nodes["Data Input.008"].inputs[0], 'value',
                 text='Dataset')

        # Shape - Hidden Layers
        col = self.layout.column(align=True)
        col.label(text='Shape - Hidden Layer')
        col.prop(AN_node.nodes["Data Input.016"].inputs[0], 'value',
                 text='Hidden Layer')

        # Output Shape
        col = self.layout.column(align=True)
        col.label(text='Output Shape')
        col.prop(AN_node.nodes["Data Input.022"].inputs[0], 'value',
                 text='Dataset')

# VIEW3D_PT_NN_model_training where the meaning of VIEW3D -> 3D view, P -> panel, T -> type and then the - name of the class


class VIEW3D_PT_NN_model_training(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'NN'
    bl_label = "Training Model"

    def draw(self, context):
        AN_node = bpy.data.node_groups["Artificial Neural Network"]

        # --------------------------------------------------------------------------------------------
        # ----------------------------------------- TRAINING -----------------------------------------
        # --------------------------------------------------------------------------------------------
        # Path of the dataset
        col = self.layout.column(align=True)
        col.label(text='Dataset Path')
        col.prop(AN_node.nodes["Data Input.005"].inputs[0], 'value',
                 text='Full Path',
                 icon='FILEBROWSER')

#        # Network Setup
#        col = self.layout.column(align=True)
#        col.label(text='Network Setup')
#        col.prop(AN_node.nodes["Data Input.007"].inputs[0], 'value',
#                 text='Epochs')
#        col.prop(AN_node.nodes["Data Input.012"].inputs[0], 'value',
#                 text='Minibatch')
#        col.prop(AN_node.nodes["Data Input.013"].inputs[0], 'value',
#                 text='Learning Rate')

#        # Operator for training
#        col = self.layout.box()
#        col.label(text='Start Training')
#        col.operator('an.execute_tree',
#                     text='Create Neural Network',
#                     icon='DOT')

#        # Create Model
#        col = self.layout.box()
#        col.label(text='Widget Modify Model')
#        col.operator('node.neurons',
#                     text='Neurons',
#                     icon='NETWORK_DRIVE')


def register():
    bpy.utils.register_class(NEURONS_OT_model)
    bpy.utils.register_class(VIEW3D_PT_NN_model_aspect)
    bpy.utils.register_class(VIEW3D_PT_NN_model_size)
    bpy.utils.register_class(VIEW3D_PT_NN_model_shape)
    bpy.utils.register_class(VIEW3D_PT_NN_model_training)


def unregister():
    bpy.utils.unregister_class(NEURONS_OT_model)
    bpy.utils.unregister_class(VIEW3D_PT_NN_model_aspect)
    bpy.utils.unregister_class(VIEW3D_PT_NN_model_size)
    bpy.utils.unregister_class(VIEW3D_PT_NN_model_shape)
    bpy.utils.unregister_class(VIEW3D_PT_NN_model_training)


# if __name__ == "__main__":
#     register()
