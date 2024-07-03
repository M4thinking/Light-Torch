import yaml
import importlib
import datetime
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Union, Callable, Any
from tqdm import tqdm
from pathlib import Path

from .utils.callbacks import Callback

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        metrics: Optional[Dict[str, callable]] = None,
        callbacks: Optional[Dict[str, Callback]] = None,
        log_dir: str = "runs"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metrics = metrics or {}
        self.callbacks = callbacks or {}
        now = datetime.datetime.now()
        self.log_dir = Path(log_dir) / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.stop_training = False

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        inputs, targets = batch["inputs"].to(self.device), batch["targets"].to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), **self._compute_metrics(outputs, targets, "train")}

    def valid_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        inputs, targets = batch["inputs"].to(self.device), batch["targets"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        return {"loss": loss.item(), **self._compute_metrics(outputs, targets, "valid")}

    def _compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, prefix: str) -> Dict[str, float]:
        return {name: metric(outputs, targets).item() for name, metric in self.metrics.items()}

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        valid_freq: float = 1.0
    ):
        train_steps = len(train_loader)
        valid_steps = int(train_steps * valid_freq)

        for callback in self.callbacks.values():
            callback.on_train_begin(self)

        for epoch in range(epochs):
            if self.stop_training:
                break

            for callback in self.callbacks.values():
                callback.on_epoch_begin(self, epoch)

            train_metrics = []
            valid_metrics = []

            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for step, batch in enumerate(train_iterator):
                for callback in self.callbacks.values():
                    callback.on_batch_begin(self, batch)

                step_metrics = self.train_step(batch)
                train_metrics.append(step_metrics)

                for callback in self.callbacks.values():
                    callback.on_batch_end(self, batch, step_metrics['loss'])

                if (step + 1) % valid_steps == 0 or (step + 1) == train_steps:
                    valid_iterator = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", leave=False)
                    for valid_batch in valid_iterator:
                        valid_step_metrics = self.valid_step(valid_batch)
                        valid_metrics.append(valid_step_metrics)

            epoch_metrics = self._log_metrics(train_metrics, valid_metrics, epoch)

            for callback in self.callbacks.values():
                callback.on_epoch_end(self, epoch, epoch_metrics)

        for callback in self.callbacks.values():
            callback.on_train_end(self)

    def _log_metrics(self, train_metrics: List[Dict[str, float]], valid_metrics: List[Dict[str, float]], epoch: int) -> Dict[str, float]:
        train_avg = {k: sum(m[k] for m in train_metrics) / len(train_metrics) for k in train_metrics[0]}
        valid_avg = {k: sum(m[k] for m in valid_metrics) / len(valid_metrics) for k in valid_metrics[0]}

        for name, value in train_avg.items():
            self.writer.add_scalar(f"train/{name}", value, epoch)
        for name, value in valid_avg.items():
            self.writer.add_scalar(f"valid/{name}", value, epoch)

        epoch_metrics = {**{f"train_{k}": v for k, v in train_avg.items()},
                         **{f"valid_{k}": v for k, v in valid_avg.items()}}

        print(f"Epoch {epoch+1} - Train: {train_avg}, Valid: {valid_avg}")
        return epoch_metrics

    def predict(self, inputs: Union[torch.Tensor, DataLoader]) -> torch.Tensor:
        self.model.eval()
        
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                return self.model(inputs)
        elif isinstance(inputs, DataLoader):
            all_predictions = []
            for batch in tqdm(inputs, desc="Predicting"):
                batch_inputs = batch["inputs"].to(self.device)
                with torch.no_grad():
                    batch_predictions = self.model(batch_inputs)
                all_predictions.append(batch_predictions)
            return torch.cat(all_predictions, dim=0)
        else:
            raise ValueError("Input must be either a torch.Tensor or a DataLoader")

    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath}")

class ConfigBuilder:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def build(self) -> Dict[str, Any]:
        # Construir modelo y optimizador por separado para manejar las dependencias entre ellos
        self.config['model'] = self._build_object(self.config['model'])
        self.config['optimizer']['args']['params'] = self.config['model'].parameters()
        self.config['optimizer'] = self._build_object(self.config['optimizer'])
        
        # Construir el resto de la configuración
        for key, value in self.config.items():
            if key not in ['model', 'optimizer']:
                self.config[key] = self._build_object(value)
        
        # Metricas en el device correcto
        for metric_name, metric_func in self.config['metrics'].items():
            self.config['metrics'][metric_name] = metric_func.to(self.config['device'])
            
        # Modelo en el device correcto
        self.config['model'].to(self.config['device'])

        return self.config

    def _build_object(self, config: Dict[str, Any]) -> Any:
        if isinstance(config, dict):
            if 'class' in config:
                class_path = config['class']
                built_args = config.get('args', {})

                # Handle special case for params evaluation
                for key, value in built_args.items():
                    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                        expr = value[2:-1].strip()  # Remove ${ and }
                        try:
                            # Evaluate the expression
                            evaluated_value = eval(expr)
                            built_args[key] = evaluated_value
                        except Exception as e:
                            raise ValueError(f"Error evaluating expression {expr}: {e}")

                module_name, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                
                return cls(**built_args) #if isinstance(cls, type) else cls
            
            else:
                return {k: self._build_object(v) if isinstance(v, dict) else v for k, v in config.items()}
        else:
            return config