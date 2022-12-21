import torch

from oml.interfaces.models import IPairwiseModel
from oml.interfaces.postprocessor import IPostprocessor


class PairwisePostprocessor(IPostprocessor):

    def __init__(self, pairwise_model: IPairwiseModel, top_n: int):
        self.model = pairwise_model
        self.top_n = top_n

    def process(self, embeddings, distance_matrix, is_query, is_gallery, mask_to_ignore):
        distance_matrix[mask_to_ignore] = float("inf")  # todo

        ids_queries_all = torch.nonzero(is_query).squeeze()
        ids_galleries_all = torch.nonzero(is_gallery).squeeze()

        n_queries = len(ids_queries_all)

        # pick top n galleries for each query with the smallest distances
        _, ids_gallery_top = torch.topk(distance_matrix, k=self.top_n, largest=False)
        ids_gallery_top = ids_gallery_top.view(-1)
        picked_galleries_ids = ids_galleries_all[ids_gallery_top]  # the size is n_queries * n

        # create n_queries * n pairs of each query and related top galleries
        ids_query_range = torch.arange(n_queries).repeat_interleave(self.top_n)
        embeddings_query = embeddings[ids_queries_all[ids_query_range]]
        embeddings_gallery = embeddings[picked_galleries_ids]

        # pass all the top pairs into the model (todo: batch-inference)
        output = self.model(embeddings_query, embeddings_gallery)

        output = output.view(n_queries, self.top_n)
        picked_galleries_ids = picked_galleries_ids.view(n_queries, self.top_n)

        return output, picked_galleries_ids
