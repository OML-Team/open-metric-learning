from abc import ABC
from collections.abc import MutableMapping
from contextlib import contextmanager
from typing import Any, Dict, Hashable, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor, cdist

from oml.utils.misc import find_first_occurrences

TSingleValues = Union[int, float, np.float_, np.int_, torch.Tensor]
TSequenceValues = Union[List[float], Tuple[float, ...], np.ndarray, torch.Tensor]
TOnlineValues = Union[TSingleValues, TSequenceValues]


def take_2d(x: Tensor, indices: Tensor) -> Tensor:
    """
    Args:
        x: Tensor with the shape of ``[N, M]``
        indices: Tensor of integers with the shape of ``[N, P]``
            Note, rows in ``indices`` may contain duplicated values.
            It means that we can take the same element from ``x`` several times.

    Returns:
        Tensor of the items picked from ``x`` with the shape of ``[N, P]``

    """
    assert x.ndim == indices.ndim == 2
    assert x.shape[0] == indices.shape[0]

    n = x.shape[0]
    ii = torch.arange(n).unsqueeze(-1).expand(n, indices.shape[1])

    return x[ii, indices]


def assign_2d(x: Tensor, indices: Tensor, new_values: Tensor) -> Tensor:
    """
    Args:
        x: Tensor with the shape of ``[N, M]``
        indices: Tensor of integers with the shape of ``[N, P]``, where ``P <= M``
        new_values: Tensor with the shape of ``[N, P]``

    Returns:
        Tensor with the shape of ``[N, M]`` constructed by the following rule:
        ``x[i, indices[i, j]] = new_values[i, j]``

    """
    assert x.ndim == indices.ndim == new_values.ndim
    assert x.shape[0] == indices.shape[0] == new_values.shape[0]
    assert indices.shape == new_values.shape

    n = x.shape[0]
    ii = torch.arange(n).unsqueeze(-1).expand(n, indices.shape[1])

    x[ii, indices] = new_values

    return x


def elementwise_dist(x1: Tensor, x2: Tensor, p: int = 2) -> Tensor:
    """
    Args:
        x1: tensor with the shape of [N, D]
        x2: tensor with the shape of [N, D]
        p: degree

    Returns: elementwise distances with the shape of [N]

    """
    assert len(x1.shape) == len(x2.shape) == 2
    assert x1.shape == x2.shape

    # we need an extra dim here to avoid pairwise behaviour of torch.cdist
    if len(x1.shape) == 2:
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

    dist = cdist(x1=x1, x2=x2, p=p).view(len(x1))

    return dist


def pairwise_dist(x1: Tensor, x2: Tensor, p: int = 2) -> Tensor:
    """
    Args:
        x1: tensor with the shape of [N, D]
        x2: tensor with the shape of [M, D]
        p: degree

    Returns: pairwise distances with the shape of [N, M]

    """
    assert len(x1.shape) == len(x2.shape) == 2
    assert x1.shape[-1] == x2.shape[-1]

    return cdist(x1=x1, x2=x2, p=p)


def normalise(x: Tensor, p: int = 2) -> Tensor:
    """
    Args:
        x: A 2D tensor
        p: Specifies the exact p-norm

    Returns:
        Normalised input

    """
    assert x.ndim == 2
    xn = torch.linalg.norm(x, p, dim=1).detach()
    x = x.div(xn.unsqueeze(1))
    return x


def get_device(model: torch.nn.Module) -> str:
    return str(next(model.parameters()).device)


def _check_is_sequence(val: Any) -> bool:
    try:
        len(val)
        return True
    except Exception:
        return False


def drop_duplicates_by_ids(ids: List[Hashable], data: Tensor, sort: bool = True) -> Tuple[List[Hashable], Tensor]:
    """
    The function returns rows of data that have unique ids.
    Thus, if there are multiple occurrences of some id, it leaves the first one.

    Args:
        ids: Identifiers of data records with the length of ``N``
        data: Tensor of data records in the shape of ``[N, *]``
        sort: Set ``True`` to return unique records sorted by their ids

    Returns:
        Unique data records with their ids

    """
    assert isinstance(ids, list)
    ids_first = find_first_occurrences(ids)
    ids = [ids[i] for i in ids_first]
    data = data[ids_first]

    if sort:
        ii_permute = torch.argsort(torch.tensor(ids))
        ids = [ids[i] for i in ii_permute]
        data = data[ii_permute]

    return ids, data


@contextmanager
def temporary_setting_model_mode(model: torch.nn.Module, set_train: bool) -> torch.nn.Module:
    prev_mode = model.training
    model.train(set_train)
    yield model
    model.train(prev_mode)


class OnlineCalc(ABC):
    """
    The base class to calculate some statistics online (on the stream of values).

    """

    def __init__(self, val: Optional[TOnlineValues] = None):
        self.result: float = 0.0
        if val is not None:
            self.update(val)

    def update(self, val: TOnlineValues) -> None:
        if _check_is_sequence(val):
            self.calc_with_batch(val)
        else:
            self.calc_with_single_value(val)

    def calc_with_single_value(self, val: TSingleValues) -> None:
        """
        Calculation with non iterable types: float, int / numpy and torch elements (array and elements of
        array have different types and methods)

        """
        self.calc_with_batch([val])

    def calc_with_batch(self, val: TSequenceValues) -> None:
        """
        Vectorized calculation of online value

        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.__dict__)})"


class AvgOnline(OnlineCalc):
    def __init__(self, *args: Any, **kwargs: Any):
        self.n = 0
        super().__init__(*args, **kwargs)

    def calc_with_batch(self, val: TSequenceValues) -> None:
        len_val = len(val)
        self.n += len_val
        self.result = sum(val) / self.n + (self.n - len_val) / self.n * self.result  # type: ignore


class SumOnline(OnlineCalc):
    def calc_with_batch(self, val: TSequenceValues) -> None:
        self.result += float(sum(val))


class OnlineDict(MutableMapping):  # type: ignore
    """
    We don't inherite from built-in 'dict' due to internal C optimization. We mimic to dict with MutableMapping
    https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/

    """

    online_calculator: Type[OnlineCalc]

    def __init__(self, input_dict: Optional[Dict[Hashable, TOnlineValues]] = None):
        self.dict: Dict[Hashable, OnlineCalc] = {}

        if input_dict:
            self.update(input_dict)

    def __setitem__(self, key: Hashable, value: TOnlineValues) -> None:
        self.dict[key] = self.online_calculator(value)

    def __getitem__(self, key: Hashable) -> float:
        return self.dict[key].result

    def __len__(self) -> int:
        return len(self.dict)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.dict)

    def __delitem__(self, key: Hashable) -> None:
        return self.dict.__delitem__(key)

    def update(self, input_dict: Dict[Hashable, TOnlineValues]) -> None:  # type: ignore
        for k, v in input_dict.items():
            if k in self.dict:
                self.dict[k].update(v)
            else:
                self[k] = v

    def __repr__(self) -> str:
        output = {k: v.result for k, v in self.dict.items()}
        return f"{self.__class__.__name__}({output})"

    def get_dict_with_results(self) -> Dict[str, float]:
        return {k: v for k, v in self.items()}


class OnlineAvgDict(OnlineDict):
    online_calculator = AvgOnline


class OnlineSumDict(OnlineDict):
    online_calculator = SumOnline


class PCA:
    """
    Principal component analysis (PCA).

    Estimate principal axes, and perform vectors transformation.

    Note:
        The code is almost the same as one from sklearn, but we had two reasons to have our own implementation.
        First, we need to work with Torch tensors instead of NumPy arrays. Second, we wanted to avoid one more external
        dependency.

    Attributes:
        components: Matrix of shape ``[embeddings_dim, embeddings_dim]``. Principal axes in embeddings space,
            representing the directions of maximum variance in the data. Equivalently, the right singular
            vectors of the centered input data, parallel to its eigenvectors. The components are sorted by
            ``explained_variance``.
        explained_variance: Array of size ``embeddings_dim``
            The amount of variance explained by each of the selected components.
            The variance estimation uses ``n_embeddings - 1`` degrees of freedom.
            Equal to  eigenvalues of the covariance matrix of ``embeddings``.
        explained_variance_ratio: Array of size ``embeddings_dim``.
            Percentage of variance explained by each of the components.
        singular_values: Array of size ``embeddings_dim``.
            The singular values corresponding to each of the selected components.
        mean: Array of size ``embeddings_dim``.
            Per-feature empirical mean, estimated from the training set.
            Equal to ``embeddings.mean(dim=0)``.

    For an embeddings matrix :math:`X` of shape :math:`n\\times d` the principal axes could be found by
    performing Singular Value Decomposition

    .. math::
        X = U\\Sigma V^T

    where :math:`U` is an :math:`n\\times n` orthogonal matrix, :math:`\\Sigma` is an :math:`n\\times d` rectangular
    diagonal matrix with non-negative real numbers on the diagonal, :math:`V` is an :math:`d\\times d` orthogonal
    matrix.

    Rows of the :math:`V` form an orthonormal basis, and could be used to project embeddings to a new space, possible
    of lower dimension:

    .. math::
        X' = X\\cdot V^T

    The inverse transform is done by

    .. math::
        X = X'\\cdot V

    See:

        `Principal Components Analysis`_

    .. _`Principal Components Analysis`:
        https://en.wikipedia.org/wiki/Principal_component_analysis

    Example:
        >>> embeddings = torch.rand(100, 5)
        >>> pca = PCA(embeddings)
        >>> embeddings_transformed = pca.transform(embeddings)
        >>> embeddings_recovered = pca.inverse_transform(embeddings_transformed)
        >>> torch.all(torch.isclose(embeddings, embeddings_recovered, atol=1.e-6))
        tensor(True)
    """

    components: torch.Tensor
    mean: torch.Tensor
    singular_values: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_ratio: torch.Tensor

    def __init__(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings: Embeddings matrix with the shape of ``[n_embeddings, embeddings_dim]``.

        """
        self._fit(embeddings)

    def _fit(self, embeddings: torch.Tensor) -> None:
        """
        Perform the PCA. Evaluate ``components``, ``expoained_variance``, ``explained_variance_ratio``,
        ``singular_values``, ``mean``.

        Args:
            embeddings: Embeddings matrix with the shape of ``[n_embeddings, embeddings_dim]``.
        """
        # The code below mirrors code from scikit-learn repository:
        # https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/decomposition/_pca.py#L491
        n_samples = embeddings.shape[0]
        self.mean = embeddings.mean(dim=0).unsqueeze(0)
        embeddings = embeddings - self.mean
        # if there are more embeddings than its dimension, then we will not perform full matrices evaluation
        full_matrices = embeddings.shape[0] < embeddings.shape[1]
        _, self.singular_values, self.components = torch.linalg.svd(embeddings, full_matrices=full_matrices)
        self.explained_variance = self.singular_values**2 / (n_samples - 1)
        self.explained_variance_ratio = self.explained_variance / self.explained_variance.sum()

        # Make components deterministic.
        # Note. In sklearn this operation is done based on the U matrix of SVD decomposition, and
        # here V matrix is used. So, the components of this class and sklearn.decomposition.PCA could differ in sign.
        # See the following for details:
        # https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/decomposition/_pca.py#L520
        n_components = self.components.shape[0]
        max_abs_rows = torch.argmax(torch.abs(self.components), dim=1)
        signs = torch.sign(self.components[torch.arange(n_components), max_abs_rows])
        self.components *= signs.unsqueeze(1)

    def transform(self, embeddings: torch.Tensor, n_components: Optional[int] = None) -> torch.Tensor:
        """
        Apply fitted PCA to transform embeddings.

        Args:
            embeddings: Matrix of shape ``[n_embeddings, embeddings_dim]``.
            n_components: The desired dimension of the output.

        Returns:
            Transformed embeddings.
        """
        if not n_components:
            n_components = embeddings.shape[1]
        self._check_dimensions(n_components)
        embeddings_ = embeddings.to(self.mean) - self.mean
        return torch.matmul(embeddings_, self.components[:n_components, :].T).to(embeddings)

    def inverse_transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse transform to embeddings.

        Args:
            embeddings: Matrix of shape ``[n_embeddings, N]`` where ``N <= embeddings_dim`` is the dimension of
                        embeddings.

        Returns:
            Embeddings projected into original embeddings space.
        """
        n_components = embeddings.shape[1]
        self._check_dimensions(n_components)
        return torch.matmul(embeddings, self.components[:n_components, :]) + self.mean

    def calc_principal_axes_number(self, pfc_variance: Tuple[float, ...]) -> torch.Tensor:
        """
        Function estimates the number of principal axes that are required to explain the `explained_variance_ths`
        variance.

        Args:
            pfc_variance: Values in range [0, 1]. Find the number of components such that the amount
                          of variance that needs to be explained is greater than the fraction specified
                          by ``pfc_variance``.
        Returns:
            List of amount of principal axes.

        Let :math:`X` be a set of :math:`d` dimensional embeddings.
        Let :math:`\\lambda_1, \\ldots, \\lambda_d\\in\\mathbb{R}` be a set of eigenvalues
        of the covariance matrix of :math:`X` sorted in descending order.
        Then for a given value of desired explained variance :math:`r`,
        the number of principal components that explaines :math:`r\\cdot 100\\%%` variance is the largest integer
        :math:`n` such that

        .. math::
            \\frac{\\sum\\limits_{i = 1}^{n - 1}\\lambda_i}{\\sum\\limits_{i = 1}^{d}\\lambda_i} \\leq r

        Example:
            In the example bellow there are 4 vectors of length 10, and only first 4 dimensions have non-zero values.
            Its covariance matrix will have only 4 eigenvalues, that are greater than 0, i.e. there are only 4 principal
            axes. So, in order to keep at least 50% of the information from the set, we need to keep 2 principal
            axes, and in order to keep all the information we need to keep 5 principal axes (one additional axis appears
            because the number of principal axes is superior to the desired explained variance threshold).

            >>> embeddings = torch.eye(4, 10, dtype=torch.float)
            >>> pca = PCA(embeddings)
            >>> pca.calc_principal_axes_number(pfc_variance=(0.5, 1))
            tensor([2, 5])

        """
        ratio_cumsum = torch.cumsum(self.explained_variance_ratio, dim=0)
        n_components = torch.searchsorted(ratio_cumsum, torch.tensor(pfc_variance), side="right") + 1
        return n_components

    def _check_dimensions(self, n_components: int) -> None:
        if n_components > self.components.shape[0]:
            raise ValueError(
                "The embeddings couldn't be transformed, due to dimensions mismatch. "
                f"Expected dimension less than or equal to {self.components.shape[0]}, but got {n_components}"
            )


__all__ = [
    "elementwise_dist",
    "pairwise_dist",
    "OnlineCalc",
    "AvgOnline",
    "SumOnline",
    "OnlineDict",
    "OnlineAvgDict",
    "OnlineSumDict",
    "take_2d",
    "assign_2d",
    "PCA",
    "drop_duplicates_by_ids",
    "normalise",
]
