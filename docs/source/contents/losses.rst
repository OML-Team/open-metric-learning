Losses
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

TripletLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.losses.triplet.TripletLoss
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward
    .. autoproperty:: last_logs


TripletLossPlain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.losses.triplet.TripletLossPlain
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward
    .. autoproperty:: last_logs

TripletLossWithMiner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.losses.triplet.TripletLossWithMiner
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward
    .. autoproperty:: last_logs

SurrogatePrecision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.losses.surrogate_precision.SurrogatePrecision
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward

ArcFaceLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.losses.arcface.ArcFaceLoss
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. autoproperty:: last_logs

ArcFaceLossWithMLP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.losses.arcface.ArcFaceLossWithMLP
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. autoproperty:: last_logs

label_smoothing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: oml.functional.label_smoothing.label_smoothing
