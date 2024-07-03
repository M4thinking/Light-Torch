class Callback:
    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch, metrics):
        pass

    def on_batch_begin(self, trainer, batch):
        pass

    def on_batch_end(self, trainer, batch, loss):
        pass

class ModelCheckpoint(Callback):
    def __init__(self, filepath, save_best_only=False, monitor='val_accuracy', mode='min', period=1):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.mode = mode
        self.monitor = monitor
        self.period = period
        self.epochs_since_last_save = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, trainer, epoch, metrics):
        self.epochs_since_last_save += 1
        save_dir = trainer.log_dir / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = save_dir / self.filepath.format(epoch=epoch + 1, **metrics)
            if self.save_best_only:
                current = metrics.get(self.monitor, None)
                if current is not None:
                    if (self.mode == 'min' and current < self.best_score) or \
                       (self.mode == 'max' and current > self.best_score):
                        self.best_score = current
                        # Borrar checkpoints anteriores que termina en .pth
                        for file in save_dir.glob("*.pth"): file.unlink()
                        trainer.save_checkpoint(filepath)
            else:
                trainer.save_checkpoint(filepath)


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, mode='min'):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer, epoch, metrics):
        current = metrics[self.monitor]
        if (self.mode == 'min' and current < self.best_score - self.min_delta) or \
           (self.mode == 'max' and current > self.best_score + self.min_delta):
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True