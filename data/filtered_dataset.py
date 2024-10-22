import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageOps

from torchvision.datasets import VisionDataset
# import cv2


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    images_per_class: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            num_images = 0
            for fname in sorted(fnames):
                if num_images >= images_per_class:
                    break
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                    num_images += 1

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        images_per_class: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file, images_per_class=images_per_class)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        images_per_class: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, images_per_class=images_per_class)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def pil_loader_single(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class FilteredImageNetDataset(DatasetFolder):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            transform_mask: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            caption=False,
            images_per_class: int = 10000,
            expansion_mask_pixels=None,
    ):
        """
        root: path to the filtered dataset containing images
        Data Structure:
        root

        ├── images
        │   ├── n01440764
        │   │   ├── n01440764_10026.JPEG

        ├── masks
        │   ├── n01440764
        │   │   ├── n01440764_10026.JPEG

        ├── captions
        │   ├── n01440764
        │   │   ├── n01440764_10026.JPEG


        """
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            images_per_class=images_per_class,
        )
        self.imgs = self.samples
        self.transform_mask = transform_mask
        self.caption = caption
        self.expansion_mask_pixels = expansion_mask_pixels
        print("Number of images: ", len(self.imgs))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image_path, target = self.samples[index]

        # mask_path = image_path.replace("images", "masks")
        # caption_path = image_path.replace("images", "captions")
        # img_extension = os.path.splitext(image_path)[1]
        # caption_path = caption_path.replace(img_extension, ".txt")

        # if self.caption:
        #     with open(caption_path, "r") as f:
        #         caption = f.read()
        #         caption = caption.split("\n")[0]
        # else:
        #     caption = "XXXX"


        image = self.loader(image_path)
        # mask = "No Mask" if not self.transform_mask else self.loader(mask_path)

        # if self.expansion_mask_pixels is not None and mask != "No Mask":
        #     mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #     # Invert the mask
        #     mask_ = cv2.bitwise_not(mask_)

        #     # Create a structuring element (kernel) for dilation
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.expansion_mask_pixels + 1,
        #                                                            2 * self.expansion_mask_pixels + 1))

        #     # Perform dilation on the inverted mask
        #     expanded_inverted_mask = cv2.dilate(mask_, kernel, iterations=1)

        #     # Invert the expanded mask back to get the desired result
        #     expanded_mask = cv2.bitwise_not(expanded_inverted_mask)
        #     mask = Image.fromarray(expanded_mask)

        if self.transform is not None:
            image = self.transform(image)
            if self.transform_mask is not None:
                mask = self.transform_mask(mask)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return image, mask, target, caption, image_path, mask_path, caption_path
        return image,target


    def __len__(self) -> int:
        return len(self.samples)


import glob
class FilteredCOCODataset(VisionDataset):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            transform_mask: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            caption=False,
            images_per_class=None,
            expansion_mask_pixels=None,
    ):
        """
        root: path to the filtered dataset containing images, masks, captions and classes folders
        Data Structure:
        root

        ├── images
        │   ├── 000000000139.jpg

        ├── masks
        │   ├── 000000000139.png

        ├── captions
        │   ├── 000000000139.txt

        ├── classes
        │  ├── 000000000139.txt

        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = glob.glob(os.path.join(root, "images/*"))
        self.transform_mask = transform_mask
        self.caption = caption
        self.loader = pil_loader
        self.expansion_mask_pixels = expansion_mask_pixels
        print("Number of images: ", len(self.imgs))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image_path = self.imgs[index]

        mask_path = image_path.replace("images", "masks")
        caption_path = image_path.replace("images", "captions")
        img_extension = os.path.splitext(image_path)[1]
        caption_path = caption_path.replace(img_extension, ".txt")

        if self.caption:
            with open(caption_path, "r") as f:
                caption = f.read()
                caption = caption.split("\n")[0]
        else:
            caption = "XXXX"


        image = self.loader(image_path)
        mask = "No Mask" if not self.transform_mask else self.loader(mask_path)

        if self.expansion_mask_pixels is not None and mask != "No Mask":
            mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Create a structuring element (kernel) for dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.expansion_mask_pixels + 1,
                                                                   2 * self.expansion_mask_pixels + 1))

            # Perform dilation on the inverted mask
            expanded_inverted_mask = cv2.dilate(mask_, kernel, iterations=1)

            # Invert the expanded mask back to get the desired result
            expanded_mask = cv2.bitwise_not(expanded_inverted_mask)
            mask = Image.fromarray(expanded_mask)

        if self.transform is not None:
            image = self.transform(image)
            if self.transform_mask is not None:
                mask = ImageOps.invert(mask) if not self.expansion_mask_pixels else mask
                mask = self.transform_mask(mask)

        target = 0

        return image, mask, target, caption, image_path, mask_path, caption_path

    def __len__(self) -> int:
        return len(self.imgs)



if __name__ == "__main__":
    # create tranform which convert image to tensor and resie to 512
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    root = "/home/hashmat/Downloads/Coco_2017/filtered"
    dataset = FilteredCOCODataset(root, transform=transform, transform_mask=transform_mask, expansion_mask_pixels=15)
    # load dataloader
    import torch
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)

    for i, (image, mask, target, caption, image_paths, mask_paths, caption_paths) in enumerate(dataloader):
        print(image.shape)
        break