import logging


# Function for getting a console logger that prints `time LEVEL message`
def get_logger(name="cyclops"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# Function for starting a W&B run if enabled in the config, else returning None
def init_wandb(cfg):
    wb = cfg["logging"]["wandb"]
    if not wb["enabled"]:
        return None

    import wandb

    return wandb.init(
        project=wb["project"],
        entity=wb["entity"],
        name=cfg["experiment"]["name"],
        config=cfg,
    )
