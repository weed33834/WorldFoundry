import os
import random
import time
from copy import deepcopy

import torch
from webdataset.utils import pytorch_worker_info

try:
    from megatron.core import parallel_state
except ImportError:
    parallel_state = None


class ParallelSyncMultiAspectRatioMixin:
    def _obtain_node_worker_url_mapping(
        self,
        url_aspect_split,
        num_urls_per_worker: int,
        group_id: int,
        group_num: int,
        worker_id: int,
        num_workers: int,
    ):
        assert self.split_by_node is True and self.split_by_worker is True

        chunk_mappings = []
        for aspect_ratio in url_aspect_split:
            samples_asp = url_aspect_split[aspect_ratio]
            nchunks_asp = int(len(samples_asp) / num_urls_per_worker)
            for chunk_id in range(nchunks_asp):
                chunk_mappings.append((aspect_ratio, samples_asp[chunk_id::nchunks_asp]))

        chunk_mappings = chunk_mappings[group_id::group_num]
        chunk_mappings = chunk_mappings[worker_id::num_workers]

        assert len(chunk_mappings) == 1, f"Length of chunk_mappings {len(chunk_mappings)} != 1"
        return chunk_mappings[0][1]

    def enable_parallel(self):
        if parallel_state is None:
            raise ImportError("megatron.core.parallel_state is required for parallel shard synchronization")
        self.group_id = parallel_state.get_data_parallel_rank()
        self.group_size = torch.distributed.get_world_size() // parallel_state.get_data_parallel_world_size()

    def obtain_url_list(self):
        rank, world_size, worker_id, num_workers = pytorch_worker_info()
        num_groups = world_size // self.group_size

        if self.resume_flag:
            self.epoch = int(os.environ.get("WDS_EPOCH_NUM", 0))
            self.start_index = int(os.environ.get("WDS_START_INDEX", 0)) // self.chunk_size

        url_aspect_split = deepcopy(self.url_aspect_split)
        nworkers_all = num_groups * num_workers

        if self.verbose:
            self.log.info(f"Total {nworkers_all} workers are in effect")

        url_aspect_split, num_urls_per_worker = self._ddp_equalize(url_aspect_split, nworkers_all)
        urls = self._obtain_node_worker_url_mapping(
            url_aspect_split,
            num_urls_per_worker,
            self.group_id,
            num_groups,
            worker_id,
            num_workers,
        )

        if self.shuffle:
            random.Random(self.group_id).shuffle(urls)

        start_index_per_worker = self.start_index // num_workers
        if not self.is_infinite_loader:
            urls = urls[start_index_per_worker:]

        if self.verbose:
            self.log.info(
                f"Rank {rank}, group {self.group_id}, worker {worker_id} of {num_workers}, "
                f"group_size {self.group_size} got {len(urls)} urls, first five are {urls[:5]}"
            )

        return urls

    def __iter__(self):
        url_list = self.obtain_url_list()

        if self.is_infinite_loader:
            while True:
                cur_time = time.time_ns()
                random.Random(cur_time).shuffle(url_list)
                for url in url_list:
                    yield dict(url=url)
        else:
            for url in url_list:
                yield dict(url=url)
