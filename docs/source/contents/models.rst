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
.. autoclass:: oml.models.meta.projection.ExtractorWithMLP
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

LinearTrivialDistanceSiamese
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.meta.siamese.LinearTrivialDistanceSiamese
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward

TrivialDistanceSiamese
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.meta.siamese.TrivialDistanceSiamese
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward

ConcatSiamese
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.meta.siamese.ConcatSiamese
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward
    .. automethod:: predict
    .. automethod:: freeze
    .. automethod:: unfreeze
