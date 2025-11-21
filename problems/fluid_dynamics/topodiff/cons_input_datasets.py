import os
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, deterministic=True
):
    """
    Create the generator used for sampling.
    The dataset should contain:
    - the .npy files of physical fields in the form cons_pf_array_X.npy,


    :param data_dir: the dataset directory.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_input_constraints = _list_input_files_recursively(data_dir)
    dataset = InputConstraintsDataset(
        all_input_constraints,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_input_files_recursively(data_dir):
    input_constraints = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            input_constraints.append(full_path) # Physical fields file
        elif os.path.isdir(full_path):
            input_constraints.extend(_list_input_files_recursively(full_path))
    return input_constraints


class InputConstraintsDataset(Dataset):
    def __init__(self, input_constraints_paths, shard=0, num_shards=1):
        super().__init__()
        self.local_input_constraints = input_constraints_paths[shard:][::num_shards]

    def __len__(self):
        return len(self.local_input_constraints)

    def __getitem__(self, idx):
        input_constraints_path = self.local_input_constraints[idx]
      
        input_constraints = np.load(input_constraints_path)

        return np.transpose(input_constraints, [2, 0, 1]).astype(np.float32)