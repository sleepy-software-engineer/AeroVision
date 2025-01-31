from logger.LoggerFactory import LoggerFactory

# todo: import the models
logger = LoggerFactory.get_logger(__name__)


def vit_model():
    logger.info("Running the ViT Model with the Datasets (Training and Testing)")

    logger.info("Finished running the Tasks")


def cnn_model():
    logger.info("Running the CNN Model with the Datasets (Training and Testing)")

    logger.info("Finished running the Tasks")


if __name__ == "__main__":
    vit_model()
    cnn_model()
