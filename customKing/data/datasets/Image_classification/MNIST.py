import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
import gzip
import lzma
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from torchvision.datasets.utils import download_and_extract_archive

from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import torch.utils.data as torchdata
from customKing.data import DatasetCatalog
from customKing.data.datasets.Image_classification.MNIST_Fashion import MNIST_Fashion_train_valid_test

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path: Union[str, IO]) -> Union[IO, gzip.GzipFile]:
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        return lzma.open(path, 'rb')
    return open(path, 'rb')


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}


def read_sn3_pascalvincent_tensor(path: Union[str, IO], strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

class MNIST_train_valid_test(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            mode: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            num_valids= 6000
    ) -> None:
        super(MNIST_train_valid_test, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        if "train" in mode or "valid" in mode:
            self.train = True  # training set or test set
        elif "test" in mode:
            self.train = False

        self.mode = mode

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        if self.mode == "valid":
            self.data = self.data[-num_valids:]
            self.targets = self.targets[-num_valids:]
        if self.mode == "train":
            self.data = self.data[:-num_valids]
            self.targets = self.targets[:-num_valids]

        if self.mode == "train_before_5":
            self.data = self.data[:-num_valids]
            self.targets = self.targets[:-num_valids]
            index = self.targets < 5
            self.data = self.data[index]
            self.targets = self.targets[index]

        if self.mode == "valid_before_5":
            self.data = self.data[-num_valids:]
            self.targets = self.targets[-num_valids:]
            index = self.targets < 5
            self.data = self.data[index]
            self.targets = self.targets[index]

        if self.mode == "test_before_5":
            index = self.targets < 5
            self.data = self.data[index]
            self.targets = self.targets[index]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

def load_MNIST(name,root):
    transform_train = transforms.Compose([
            # transforms.Pad([4]),
            # transforms.RandomCrop(28),
            transforms.ToTensor()
            ])
    
    transform_test = transforms.Compose([
            transforms.ToTensor()
            ])

    if name == "MNIST_train":
        dataset = MNIST_train_valid_test(root=root, mode="train", download=True, transform=transform_train) #训练数据集

    elif name == "MNIST_valid":
        dataset = MNIST_train_valid_test(root=root, mode="valid", download=True, transform=transform_test)
    elif name == "MNIST_train_and_valid":
        dataset_train = MNIST_train_valid_test(root=root, mode="train", download=True, transform=transform_train) #训练数据集
        dataset_valid = MNIST_train_valid_test(root=root, mode="valid", download=True, transform=transform_train) #训练数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid])
    if name =="MNIST_test":
        dataset = MNIST_train_valid_test(root=root, mode="test", download=True, transform=transform_test)
    if name == "MNIST_train_and_valid_and_test":
        dataset_train = MNIST_train_valid_test(root=root, mode="train", download=True, transform=transform_train) #训练数据集
        dataset_valid = MNIST_train_valid_test(root=root, mode="valid", download=True, transform=transform_train) #验证数据集
        dataset_test = MNIST_train_valid_test(root=root, mode="test", download=True, transform=transform_train) #验证数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid,dataset_test])

    if name == "MNIST_train_before_5":
        dataset = MNIST_train_valid_test(root=root, mode="train_before_5", download=True, transform=transform_train) #训练数据集
    if name == "MNIST_valid_before_5":
        dataset = MNIST_train_valid_test(root=root, mode="valid_before_5", download=True, transform=transform_train) #训练数据集
    elif name == "MNIST_train_and_valid_before_5":
        dataset_train = MNIST_train_valid_test(root=root, mode="train_before_5", download=True, transform=transform_train) #训练数据集
        dataset_valid = MNIST_train_valid_test(root=root, mode="valid_before_5", download=True, transform=transform_train) #训练数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid])
    if name =="MNIST_test_before_5":
        dataset = MNIST_train_valid_test(root=root, mode="test_before_5", download=True, transform=transform_test)

    if name == "MNIST_Digits_Fashion_test":
        Digits_dataset = MNIST_train_valid_test(root=root, mode="test", download=True, transform=transform_test)
        Fashion_dataset = MNIST_Fashion_train_valid_test(root=root,mode="test",download=True, transform=transform_test)
        Fashion_dataset.targets = [label+10 for label in Fashion_dataset.targets]
        dataset = torchdata.ConcatDataset([Digits_dataset,Fashion_dataset])
    return dataset

def register_MNIST(name,root):
    DatasetCatalog.register(name, lambda: load_MNIST(name,root))
