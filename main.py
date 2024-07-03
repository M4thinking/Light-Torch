import argparse
import torch
from src.trainer import Trainer, ConfigBuilder

def main(args):
    # Leer la configuración del archivo YAML
    config_builder = ConfigBuilder(args.config)
    config = config_builder.build()
    
    if args.list_checkpoints: # Listar todos los checkpoints en el directorio de logs
        import glob
        checkpoints = glob.glob(f"{config['log_dir']}/*.pth")
        if len(checkpoints) == 0:
            print("No checkpoints found.")

    # Instanciar el Trainer
    trainer = Trainer(
        model     = config['model'],
        optimizer = config['optimizer'],
        criterion = config['criterion'],
        device    = config['device'],
        metrics   = config['metrics'],
        callbacks = config['callbacks'],
        log_dir   = config['log_dir']
    )

    if args.from_checkpoint: # Cargar checkpoint si existe
        trainer.load_checkpoint(args.from_checkpoint)
    
    if args.train: # Entrenar el modelo
        # Obtener los datos usando la función especificada en la configuración
        train_loader, valid_loader = config_builder._build_object(config['data_loader'])
        trainer.train(train_loader, valid_loader, epochs=config['epochs'], valid_freq=config['valid_freq'])

    if args.predict: # Realizar predicciones
        # Obtener los datos usando la función especificada en la configuración
        _, valid_loader = config_builder._build_object(config['data_loader'])
        # Realizar predicciones en el conjunto de validación
        val_predictions = trainer.predict(valid_loader)

        # Calcular y mostrar las métricas finales en el conjunto de validación
        val_targets = torch.cat([batch['targets'] for batch in valid_loader]).to(config['device'])
        
        # Pasar métrics al device correcto
        for metric_name, metric_func in config['metrics'].items():
            final_metric = metric_func(val_predictions, val_targets)
            print(f"Validation {metric_name}: {final_metric:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deep learning model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--from_checkpoint', type=str, default=None, help='Path to a checkpoint file to resume training')
    parser.add_argument('--list_checkpoints', action='store_true', help='List all the checkpoints in the log directory')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions on the validation set')
    
    args = parser.parse_args()
    main(args)