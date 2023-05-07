Dataset format
~~~~~~~~~~~~~~

To reuse as much from OML as possible, you need to prepare a `.csv` file in the required format.
It's not obligatory, especially if you implement your own Dataset, but the format is required in case
of usage built-in datasets or Pipelines. You can check out the
`tiny dataset <https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4>`_
as an example.

Required columns:

* ``label`` - integer value indicates the label of item.
* ``path`` - path to image. It may be global or relative path (in these case you need to pass ``dataset_root`` to build-in Datasets.)
* ``split`` - must be one of 2 values: ``train`` or ``validation``.
* ``is_query``, ``is_gallery`` - have to be ``None`` where ``split == train`` and ``True`` (or ``1``)
  or ``False`` (or ``0``) where ``split == validation``. Note, that both values can be ``True`` at
  the same time. Then we will validate every item
  in the validation set using the "1 vs rest" approach (datasets of this kind are ``SOP``, ``CARS196`` or ``CUB``).

Optional columns:

* ``category`` - category which groups sets of similar labels (like ``dresses``, or ``furniture``).
* ``x_1``, ``x_2``, ``y_1``, ``y_2`` - integers, the format is ``left``, ``right``, ``top``, ``bot`` (``x_1`` and ``y_1`` must be less than ``x_2`` and ``y_2``).
  If only part of your images has bounding boxes, just fill the corresponding row with empty values.

Check out the
`examples <https://drive.google.com/drive/folders/12QmUbDrKk7UaYGHreQdz5_nPfXG3klNc?usp=sharing>`_
of valid dataset. You can also use helper to check if your dataset is in the right format:

.. code-block:: python

    import pandas as pd
    from oml.utils.dataframe_format import check_retrieval_dataframe_format

    check_retrieval_dataframe_format(df=pd.read_csv("/path/to/table.csv"), dataset_root="/path/to/dataset/root/")
