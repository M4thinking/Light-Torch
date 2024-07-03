#!/bin/bash
# chmod +x run_training.sh desde la terminal para dar permisos de ejecución

# Cambia al directorio del proyecto
cd "$(dirname "$0")"

# Ejecuta el script main.py con la configuración de MNIST
python main.py --config config/mnist_config.yaml --train

# ./run_training.sh desde la terminal para ejecutar el script