import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg
from customKing.data import DatasetCatalog
import torch.utils.data as torchdata
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


ARCHIVE_META = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": ("ILSVRC2012_devkit_t12.tar", "fa75699e90414af021442c21a62c3abf"),
}

META_FILE = "meta.bin"


class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        self.parse_archives()
        wnid_to_classes = self.load_meta_file(self.root)[0]

        super().__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            self.parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == "train":
                self.parse_train_archive(self.root)
            elif self.split == "val":
                self.parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


    def load_meta_file(self,root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
        if file is None:
            file = META_FILE
        file = os.path.join(root, file)

        if check_integrity(file):
            return torch.load(file)
        else:
            msg = (
                "The meta file {} is not present in the root directory or is corrupted. "
                "This file is automatically created by the ImageNet dataset."
            )
            raise RuntimeError(msg.format(file, root))


    def _verify_archive(self,root: str, file: str, md5: str) -> None:
        if not check_integrity(os.path.join(root, file), md5):
            msg = (
                "The archive {} is not present in the root directory or is corrupted. "
                "You need to download it externally and place it in {}."
            )
            raise RuntimeError(msg.format(file, root))


    def parse_devkit_archive(self,root: str, file: Optional[str] = None) -> None:
        """Parse the devkit archive of the ImageNet2012 classification dataset and save
        the meta information in a binary file.

        Args:
            root (str): Root directory containing the devkit archive
            file (str, optional): Name of devkit archive. Defaults to
                'ILSVRC2012_devkit_t12.tar.gz'
        """
        import scipy.io as sio

        def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, Tuple[str, ...]]]:
            metafile = os.path.join(devkit_root, "data", "meta.mat")
            meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
            nums_children = list(zip(*meta))[4]
            meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
            idcs, wnids, classes = list(zip(*meta))[:3]
            classes = [tuple(clss.split(", ")) for clss in classes]
            idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
            wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
            return idx_to_wnid, wnid_to_classes

        def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
            file = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
            with open(file) as txtfh:
                val_idcs = txtfh.readlines()
            return [int(val_idx) for val_idx in val_idcs]

        @contextmanager
        def get_tmp_dir() -> Iterator[str]:
            tmp_dir = tempfile.mkdtemp()
            try:
                yield tmp_dir
            finally:
                shutil.rmtree(tmp_dir)

        archive_meta = ARCHIVE_META["devkit"]
        if file is None:
            file = archive_meta[0]
        md5 = archive_meta[1]

        self._verify_archive(root, file, md5)

        with get_tmp_dir() as tmp_dir:
            extract_archive(os.path.join(root, file), tmp_dir)

            devkit_root = os.path.join(tmp_dir, "ILSVRC2012_devkit_t12")
            idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
            val_idcs = parse_val_groundtruth_txt(devkit_root)
            val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

            torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))


    def parse_train_archive(self,root: str, file: Optional[str] = None, folder: str = "train") -> None:
        """Parse the train images archive of the ImageNet2012 classification dataset and
        prepare it for usage with the ImageNet dataset.

        Args:
            root (str): Root directory containing the train images archive
            file (str, optional): Name of train images archive. Defaults to
                'ILSVRC2012_img_train.tar'
            folder (str, optional): Optional name for train images folder. Defaults to
                'train'
        """
        archive_meta = ARCHIVE_META["train"]
        if file is None:
            file = archive_meta[0]
        md5 = archive_meta[1]

        self._verify_archive(root, file, md5)

        train_root = os.path.join(root, folder)
        extract_archive(os.path.join(root, file), train_root)

        archives = [os.path.join(train_root, archive) for archive in os.listdir(train_root)]
        for archive in archives:
            extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)

    def parse_val_archive(self,
        root: str, file: Optional[str] = None, wnids: Optional[List[str]] = None, folder: str = "val"
    ) -> None:
        """Parse the validation images archive of the ImageNet2012 classification dataset
        and prepare it for usage with the ImageNet dataset.

        Args:
            root (str): Root directory containing the validation images archive
            file (str, optional): Name of validation images archive. Defaults to
                'ILSVRC2012_img_val.tar'
            wnids (list, optional): List of WordNet IDs of the validation images. If None
                is given, the IDs are loaded from the meta file in the root directory
            folder (str, optional): Optional name for validation images folder. Defaults to
                'val'
        """
        archive_meta = ARCHIVE_META["val"]
        if file is None:
            file = archive_meta[0]
        md5 = archive_meta[1]
        if wnids is None:
            wnids = self.load_meta_file(root)[1]

        self._verify_archive(root, file, md5)

        val_root = os.path.join(root, folder)
        extract_archive(os.path.join(root, file), val_root)

        images = sorted(os.path.join(val_root, image) for image in os.listdir(val_root))

        for wnid in set(wnids):
            os.mkdir(os.path.join(val_root, wnid))

        for wnid, img_file in zip(wnids, images):
            shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))

class ImageNet_train_valid_test(Dataset):
    def __init__(self,root,mode,transform=None,target_transform=None) -> None:
        super().__init__()

        if mode == "train":
            self.ImageNet_list = ImageNet(root=root, split="train") #训练数据集
            self.imgs_list = self.ImageNet_list.imgs
        else:
            self.ImageNet_list = ImageNet(root=root, split="val") #训练数据集
            self.imgs_list = self.ImageNet_list.imgs
            if mode == "valid":
                valid_imgs_list = []
                for i in range(1000):
                    valid_imgs_list = valid_imgs_list + self.imgs_list[i*50:i*50+25]
                self.imgs_list = valid_imgs_list
            elif mode == "test":
                test_imgs_list = []
                for i in range(1000):
                    test_imgs_list = test_imgs_list + self.imgs_list[i*50+25:(i+1)*50]
                self.imgs_list = test_imgs_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        image_path, target = self.imgs_list[index]
        pil_image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

def load_ImageNet(name,root):
    transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.Pad([28]),
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(), #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  #R,G,B每层的归一化用到的均值和方差
            ])
    
    transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    if name == "ImageNet_train":
        dataset = ImageNet_train_valid_test(root=root, mode="train", transform=transform_train) #训练数据集
    elif name == "ImageNet_valid":
        dataset = ImageNet_train_valid_test(root=root, mode="valid", transform=transform_test)
    elif name == "ImageNet_train_and_valid":
        dataset_train = ImageNet_train_valid_test(root=root, mode="train", transform=transform_train) #训练数据集
        dataset_valid = ImageNet_train_valid_test(root=root, mode="valid", transform=transform_train) #训练数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid])
    if name =="ImageNet_test":
        dataset = ImageNet_train_valid_test(root=root, mode="test", transform=transform_test)
    if name == "ImageNet_train_and_valid_and_test":
        dataset_train = ImageNet_train_valid_test(root=root, mode="train", transform=transform_train) #训练数据集
        dataset_valid = ImageNet_train_valid_test(root=root, mode="valid", transform=transform_train) #验证数据集
        dataset_test = ImageNet_train_valid_test(root=root, mode="test", transform=transform_train) #验证数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid,dataset_test])
    return dataset

def register_ImageNet(name,root):
    DatasetCatalog.register(name, lambda: load_ImageNet(name,root))