OML is available in PyPI:

```shell
pip install -U open-metric-learning
```

You can also pull the prepared image from DockerHub...

```shell
docker pull omlteam/oml:gpu
docker pull omlteam/oml:cpu
```

...or build one by your own

```shell
make docker_build RUNTIME=cpu
make docker_build RUNTIME=gpu
```