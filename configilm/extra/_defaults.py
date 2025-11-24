from torchvision.transforms import v2

from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.CustomTorchClasses import MyRotateTransform


def default_train_transform(img_size, mean, std):
    return v2.Compose(
        [
            v2.Resize(img_size, antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.Normalize(mean, std),
        ]
    )


def default_train_transform_with_noise(img_size, mean, std, gaussian_sigma: float = 20.0):
    """
    Returns a default training transform with gaussian noise added.
    Also contains additional rotation.
    """
    return v2.Compose(
        [
            v2.Resize(img_size, antialias=True),
            MyGaussianNoise(sigma=gaussian_sigma),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            MyRotateTransform([0, 90, 180, 270]),
            v2.Normalize(mean, std),
        ]
    )


def default_transform(img_size, mean, std):
    return v2.Compose(
        [
            v2.Resize(img_size, antialias=True),
            v2.Normalize(mean, std),
        ]
    )
