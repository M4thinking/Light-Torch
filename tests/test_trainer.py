import os
import sys
import torch
import shutil
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.trainer import Trainer

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock(spec=torch.nn.Module)
        self.optimizer = MagicMock(spec=torch.optim.Optimizer)
        self.criterion = MagicMock(spec=torch.nn.Module)
        self.device = torch.device('cpu')
        self.metrics = {'accuracy': MagicMock()}
        self.callbacks = {'checkpoint': MagicMock()}
        self.log_dir = 'runs/test'

    @patch('src.trainer.SummaryWriter')
    def test_trainer_initialization(self, mock_summary_writer):
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            metrics=self.metrics,
            callbacks=self.callbacks,
            log_dir=self.log_dir
        )
        self.assertEqual(trainer.model, self.model)
        self.assertEqual(trainer.optimizer, self.optimizer)
        self.assertEqual(trainer.criterion, self.criterion)
        self.assertEqual(trainer.device, self.device)
        self.assertEqual(trainer.metrics, self.metrics)
        self.assertEqual(trainer.callbacks, self.callbacks)
        self.assertTrue(mock_summary_writer.called)

    # Aquí puedes agregar más pruebas, como por ejemplo:
    # - test_train_method: para probar el método de entrenamiento.
    # - test_callbacks_integration: para probar la integración y el funcionamiento de los callbacks.
    # - test_checkpoint_loading: para probar la carga de checkpoints.
    
    def tearDown(self):
        if os.path.exists('runs'): # Eliminar el directorio de logs
            shutil.rmtree('runs')

    if __name__ == '__main__':
        unittest.main() # python -m unittest test_trainer.py

if __name__ == '__main__':
    unittest.main() # python -m unittest test_trainer.py