from multiprocessing import Process

from benchmark import cifar10, cifar100, stl10
from logger.LoggerFactory import LoggerFactory
from models.CNN import CNN
from models.TinyViT import TinyViT

logger = LoggerFactory.get_logger(__name__)


def run_cifar10_tinyvit():
    logger.info("Running TinyViT on CIFAR10")
    model = TinyViT(
        num_classes=10,
        embed_dim=128,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_heads=8,
        num_layers=6,
        mlp_dim=512,
    )
    cifar10.cifar_10(model, "TinyViT")
    logger.info("Finished TinyViT on CIFAR10")


def run_cifar100_tinyvit():
    logger.info("Running TinyViT on CIFAR100")
    model = TinyViT(
        num_classes=100,
        embed_dim=128,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_heads=8,
        num_layers=6,
        mlp_dim=512,
    )
    cifar100.cifar_100(model, "TinyViT")
    logger.info("Finished TinyViT on CIFAR100")


def run_stl10_tinyvit():
    logger.info("Running TinyViT on STL10")
    model = TinyViT(
        num_classes=10,
        embed_dim=128,
        img_size=96,
        patch_size=4,
        in_chans=3,
        num_heads=8,
        num_layers=6,
        mlp_dim=512,
    )
    stl10.stl_10(model, "TinyViT")
    logger.info("Finished TinyViT on STL10")


def run_cifar10_cnn():
    logger.info("Running CNN on CIFAR10")
    model = CNN(num_classes=10)
    cifar10.cifar_10(model, "CNN")
    logger.info("Finished CNN on CIFAR10")


def run_cifar100_cnn():
    logger.info("Running CNN on CIFAR100")
    model = CNN(num_classes=100)
    cifar100.cifar_100(model, "CNN")
    logger.info("Finished CNN on CIFAR100")


def run_stl10_cnn():
    logger.info("Running CNN on STL10")
    model = CNN(num_classes=10)
    stl10.stl_10(model, "CNN")
    logger.info("Finished CNN on STL10")


if __name__ == "__main__":
    processes = []

    processes.append(Process(target=run_cifar10_tinyvit))
    processes.append(Process(target=run_cifar100_tinyvit))
    processes.append(Process(target=run_stl10_tinyvit))
    processes.append(Process(target=run_cifar10_cnn))
    processes.append(Process(target=run_cifar100_cnn))
    processes.append(Process(target=run_stl10_cnn))

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    logger.info("All tasks have been completed.")
