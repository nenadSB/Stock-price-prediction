import logging
import os
import datetime

class Logger:
    """
    Handles logging for training, evaluation, and predictions.
    """
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger()

    def log(self, message):
        """
        Log a message.
        """
        print(message)
        self.logger.info(message)
