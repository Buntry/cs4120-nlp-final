from keras.callbacks import Callback
import pandas as pd

# To gather information about training
class LogBatchLoss(Callback):
    def __init__(self, filepath, log_every_x_batches=100):
        self.filepath = filepath
        self.interval = log_every_x_batches
        self.epoch = 0
        self.results = []

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.interval == 0 and logs:
            self.results.append({
                'batch': batch,
                'loss': logs['loss'],
                'epoch': self.epoch
            })
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_train_end(self, logs=None):
        pd.DataFrame(self.results).to_csv(self.filepath, index=False)
