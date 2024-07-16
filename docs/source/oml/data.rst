Dataset format
~~~~~~~~~~~~~~

To reuse as much from OML as possible, you need to prepare a `.csv` file in the required format.
It's not obligatory, especially if you implement your own Dataset, but the format is required in case
of usage built-in datasets or Pipelines. You can check out the mock
`images <https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4>`_,
`texts <https://github.com/OML-Team/open-metric-learning/blob/main/oml/utils/download_mock_dataset.py#L83>`_,
or `audio <https://drive.google.com/drive/folders/1NcKnyXqDyyYARrDETmhJcTTXegO3W0Ju>`_
datasets.

**Required columns:**

* ``label`` - integer value indicates the label of item.
* ``split`` - must be one of 2 values: ``train`` or ``validation``.
* ``is_query``, ``is_gallery`` - have to be ``None`` where ``split == train`` and ``True`` (or ``1``)
  or ``False`` (or ``0``) where ``split == validation``. Note, that both values can be ``True`` at
  the same time. Then we will validate every item
  in the validation set using the "1 vs rest" approach (datasets of this kind are ``SOP``, ``CARS196`` or ``CUB``).
* [**Images**, **Audios**] ``path`` - path to image/audio. It may be global or relative path (in these case you need to pass ``dataset_root`` to build-in Datasets.)
* [**Texts**] ``text`` - text describing an item.


**Optional columns:**

* ``category`` - category which groups sets of similar labels (like ``dresses``, or ``furniture``).
* ``sequence`` - ids of sequences of photos that may be useful to handle in Re-id tasks. Must be strings or integers. Take a look at the detailed `example <https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#handling-sequences-of-photos>`_.
* [**Images**] ``x_1``, ``x_2``, ``y_1``, ``y_2`` - integers, the format is ``left``, ``right``, ``top``, ``bot`` (``x_1`` and ``y_1`` must be less than ``x_2`` and ``y_2``). If only part of your images has bounding boxes, just fill the corresponding row with empty values.
* [**Audios**] ``start_time`` - a float representing the time offset from which the audio should start being read.

Check out the
`examples <https://drive.google.com/drive/folders/12QmUbDrKk7UaYGHreQdz5_nPfXG3klNc?usp=sharing>`_
of dataframes. You can also use helper to check if your dataset is in the right format:

.. mdinclude:: ../../../docs/readme/examples_source/retrieval_format.md
