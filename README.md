# Trainer

Este proyecto implementa un entrenador genérico para modelos de deep learning, con un ejemplo de uso en el conjunto de datos MNIST.

## Estructura del proyecto

La estructura del proyecto es la siguiente:

```bash
project_root/
│
├── config/
│   └── mnist_config.yaml
│
├── src/
│   ├── init.py
│   ├── trainer.py
│   ├── models/
│   │   ├── init.py
│   │   └── mnist_model.py
│   └── utils/
│       ├── init.py
│       └── metrics.py
│
├── data/
│   └── mnist/
│
├── main.py
├── requirements.txt
└── README.md
```

## Configuración

La configuración del entrenamiento se encuentra en `config/mnist_config.yaml`. Modifique este archivo para ajustar los parámetros del entrenamiento.

## Extensión

Para usar este entrenador con otros modelos o conjuntos de datos:

1. Crear un nuevo archivo de modelo en `src/models/`
2. Crear un nuevo archivo de configuración en `config/`
3. Modificar `main.py` para usar el nuevo modelo y configuración
