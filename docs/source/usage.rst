Usage
=====

.. _installation:

Installation
------------

OML is available in PyPI:

.. code-block:: console
    pip install -U open-metric-learning

You can also pull the prepared image from DockerHub or build one by your own

.. code-block:: console

   docker pull omlteam/oml:gpu
   docker pull omlteam/oml:cpu

    make docker_build RUNTIME=cpu
    make docker_build RUNTIME=gpu
