Python examples
==============================================================================

Train


..
    [comment]:vanilla-train-start

.. code-block:: python

    import torch
    from tqdm import tqdm

    from oml.datasets.base import DatasetWithLabels
    from oml.losses.triplet import TripletLossWithMiner
    from oml.miners.inbatch_all_tri import AllTripletsMiner
    from oml.models.vit.vit import ViTExtractor
    from oml.samplers.balance import BalanceSampler
    from oml.utils.download_mock_dataset import download_mock_dataset

    dataset_root = "mock_dataset/"
    df_train, _ = download_mock_dataset(dataset_root)

    model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)
    criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
    sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

    for batch in tqdm(train_loader):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

..
    [comment]:vanilla-train-end

Validate

..
    [comment]:vanilla-validation-start

.. code-block:: python

    import torch
    from tqdm import tqdm

    from oml.datasets.base import DatasetQueryGallery
    from oml.metrics.embeddings import EmbeddingMetrics
    from oml.models.vit.vit import ViTExtractor
    from oml.utils.download_mock_dataset import download_mock_dataset

    dataset_root =  "mock_dataset/"
    _, df_val = download_mock_dataset(dataset_root)

    model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).eval()

    val_dataset = DatasetQueryGallery(df_val, dataset_root=dataset_root)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
    calculator = EmbeddingMetrics()
    calculator.setup(num_samples=len(val_dataset))

    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch["embeddings"] = model(batch["input_tensors"])
            calculator.update_data(batch)

    metrics = calculator.compute_metrics()

..
    [comment]:vanilla-validation-end

Lightning

..
    [comment]:lightning-start

.. code-block:: python

    import pytorch_lightning as pl
    import torch

    from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
    from oml.lightning.modules.retrieval import RetrievalModule
    from oml.lightning.callbacks.metric import  MetricValCallback
    from oml.losses.triplet import TripletLossWithMiner
    from oml.metrics.embeddings import EmbeddingMetrics
    from oml.miners.inbatch_all_tri import AllTripletsMiner
    from oml.models.vit.vit import ViTExtractor
    from oml.samplers.balance import BalanceSampler
    from oml.utils.download_mock_dataset import download_mock_dataset

    dataset_root =  "mock_dataset/"
    df_train, df_val = download_mock_dataset(dataset_root)

    # model
    model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)

    # train
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)
    criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
    batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=3)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler)

    # val
    val_dataset = DatasetQueryGallery(df_val, dataset_root=dataset_root)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
    metric_callback = MetricValCallback(metric=EmbeddingMetrics())

    # run
    pl_model = RetrievalModule(model, criterion, optimizer)
    trainer = pl.Trainer(max_epochs=1, callbacks=[metric_callback], num_sanity_val_steps=0)
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

..
    [comment]:lightning-end

Models

..
    [comment]:checkpoint-start

.. code-block:: python

    import oml
    from oml.models.vit.vit import ViTExtractor

    # We are downloading vits16 pretrained on CARS dataset:
    model = ViTExtractor(weights="vits16_cars", arch="vits16", normalise_features=False)

    # You can also check other available pretrained models...
    print(list(ViTExtractor.pretrained_models.keys()))

    # ...or check other available types of architectures
    print(oml.registry.models.MODELS_REGISTRY)

    # It's also possible to use `weights` argument to directly pass the path to the checkpoint:
    model_from_disk = ViTExtractor(weights=oml.const.CKPT_SAVE_ROOT / "vits16_cars.ckpt", arch="vits16", normalise_features=False)

..
    [comment]:checkpoint-end
