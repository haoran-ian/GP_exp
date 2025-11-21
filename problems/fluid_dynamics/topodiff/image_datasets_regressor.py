import math
import random
import os

from PIL import Image
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False
):
    """
    Create the generator used for training the regressor predicting the compliance.
    The dataset should contain:
    - the .png images of the topologies in the form gt_topo_X.png.

    :param data_dir: the dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size of the images.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_images, q_dot = _list_image_files_recursively(data_dir)
    q_dot = np.load(q_dot)
    
    dataset = ImageDataset(
        image_size,
        all_images,
        q_dot=q_dot,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    images = []
    q_dot = ""
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            images.append(full_path)
        elif "." in entry and ext.lower() in ["npy"]:
            if entry == "q_dot.npy":
                q_dot = q_dot + full_path
            else:
                pass
        elif os.path.isdir(full_path):
            images.extend(_list_image_files_recursively(full_path))
    return images, q_dot


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        q_dot,
        shard=0,
        num_shards=1
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_q_dot = q_dot[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        image_path = self.local_images[idx]
        num_im = int((image_path.split("_")[-1]).split(".")[0])

        with open(image_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        #arr = center_crop_arr(pil_image, self.resolution)
        arr = resize_arr(pil_image, self.resolution)

        arr = np.mean(arr, axis = 2)
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = arr.reshape(self.resolution, self.resolution, 1)

        out_dict = {}
        out_dict["q_dot"] = np.array(self.local_q_dot[num_im], dtype=np.float32)
        return np.transpose(arr, [2, 0, 1]).astype(np.float32), out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def resize_arr(pil_image, image_size):

    pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    
    return arr