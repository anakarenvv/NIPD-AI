import tensorflow as tf
from tensorflow.data.experimental import service

# Paso 1: Crear un dataset simple
def preprocess(x):
    tf.print("Procesando:", x)
    return x * x

dataset = tf.data.Dataset.range(10).map(preprocess)

# Paso 2: Iniciar el servicio (dispatcher + worker)
dispatcher = service.start_dispatch_server()
worker = service.start_worker_server(dispatcher_address=dispatcher.target)

# Paso 3: Aplicar distribución del dataset
distributed_dataset = dataset.apply(
    tf.data.experimental.service.distribute(
        processing_mode="parallel_epochs",
        service=dispatcher.target
    )
)

# Paso 4: Iterar para probar
for item in distributed_dataset:
    print("Resultado:", item.numpy())

