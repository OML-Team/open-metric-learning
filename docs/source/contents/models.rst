Models
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

ViTExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.vit.vit.ViTExtractor
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: draw_attention
    .. autoproperty:: feat_dim

ViTCLIPExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.vit.clip.ViTCLIPExtractor
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. autoproperty:: feat_dim

ResnetExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.resnet.ResnetExtractor
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: draw_gradcam
    .. autoproperty:: feat_dim

ExtractorWithMLP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.projection.ExtractorWithMLP
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

vit_with_mlp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: oml.models.projection.vit_with_mlp

vit_clip_with_mlp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: oml.models.projection.vit_clip_with_mlp

resnet_with_mlp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: oml.models.projection.resnet_with_mlp
