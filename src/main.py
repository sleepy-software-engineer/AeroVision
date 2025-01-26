from core.cifar100 import cifar_100
from logger.LoggerFactory import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


def main():
    logger.info("Running the Dataset")
    # cifar_10()
    # stl_10()
    cifar_100()
    logger.info("Finished running the Dataset")


if __name__ == "__main__":
    main()
