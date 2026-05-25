from typing import Optional, Union
from dataclasses import dataclass

import torch
from torch import Tensor
from tensordict import TensorDict
import numpy as np

@dataclass
class Batch:
    tensors: TensorDict
    metadata: dict[str, list]


    def __len__(self) -> int:
        return self.tensors.batch_size[0]

    @classmethod
    def from_dict(cls, tensor_dict: dict[str, Tensor], metadata: dict[str, list]):
        batch_size = None
        for k, v in tensor_dict.items():
            if batch_size is None:
                batch_size = v.shape[0]
                pivot_key = k
            else:
                current_batch_size = v.shape[0]
                assert batch_size == current_batch_size, (
                    f"Not all tensors in tensors have the same batch size. "
                    f"{pivot_key} has {batch_size}, {k} has {current_batch_size}"
                )
        
        tensors = TensorDict(tensor_dict, batch_size=batch_size)

        for k, v in metadata.items():
            if isinstance(v, np.ndarray) and v.shape[0] != batch_size:
                raise ValueError(f"metadata '{k}' has {v.shape[0]} rows, expected {batch_size}")
        return cls(tensors=tensors, metadata=metadata)


    def chunk(self, n: int) -> list["Batch"]:
        # TODO(rafa): we assume for now tensors can be evenly divided
        tensors_list = self.tensors.chunk(chunks=n, dim=0)
        bsz_in_list = np.array([tensors.batch_size[0] for tensors in tensors_list])
        chunk_indices = np.cumsum(bsz_in_list)[:-1]

        metadata_batch_list = [{} for _ in range(n)]
        for k, v in self.metadata.items():
            assert isinstance(v, np.ndarray)

            metadata_list = np.array_split(v, chunk_indices.tolist())
            assert len(metadata_list) == n
            for i in range(n):
                metadata_batch_list[i][k] = metadata_list[i]
        
        output = []
        for i in range(n):
            output.append(
                Batch(tensors=tensors_list[i], metadata=metadata_batch_list[i])
            )
        return output

    @staticmethod
    def cat(batches: list["Batch"]):
        tensors = torch.cat([b.tensors for b in batches], dim=0)
        metadata = {}
        for k in batches[0].metadata:
            metadata[k] = np.concatenate([b.metadata[k] for b in batches], axis=0)
        return Batch(tensors=tensors, metadata=metadata)
    
    def union(self, other: "Batch"):
        for k in other.tensors.keys():
            if k in self.tensors.keys() and not torch.equal(self.tensors[k], other.tensors[k]):
                raise ValueError(f"Key conflict in tensors: {k}")
        for k in other.metadata:
            if k in self.metadata and not np.array_equal(self.metadata[k], other.metadata[k]):
                raise ValueError(f"Key conflict in metadata: {k}")
        
        self.tensors.update(other.tensors)
        self.metadata.update(other.metadata)
        return self

    def repeat_interleave(self, n: int):
        return Batch(
            tensors=self.tensors.repeat_interleave(n, dim=0),
            metadata={k: np.repeat(v, n, axis=0) for k, v in self.metadata.items()}
        )

    def __getitem__(self, idx: int):
        return Batch(
            tensors=self.tensors[idx],
            metadata={k: v[idx] for k, v in self.metadata.items()},
        )

    def select(self, tensor_keys: Optional[list[str]] = None, metadata_keys: Optional[list[str]] = None):
        tensors = self.tensors.select(*tensor_keys if tensor_keys else self.tensors)
        metadata = {k: v for k, v in self.metadata.items() if k in metadata_keys} if metadata_keys else self.medata
        return Batch(tensors=tensors, metadata=metadata)

    def to(self, device: Union[torch.Device, str]):
        self.tensors = self.tensors.to(device)
        return self

    

    


                    

            




