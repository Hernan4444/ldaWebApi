# LDA Web Api

Código usado para la api de LDA

# Endspoint

### /pin

Debido a que esta api está montada en un servidor de Heroku gratuita, este se duerme cada cierto tiempo. Esta ruta es utilizada para encender el servidor al momento de acceder la plataforma web y descargar las bases de datos.

### /lda

Ruta para ejecutar LDA. Los argumentos para esta ruta son:

* token: Llave secreta que permite el acceso a la api
* iterations: Número de iteraciones para ejecutar LDA
* mode: Modo para ejcutar LDA. Este puede ser "LDA", "Seeded LDA", "Interactive LDA" o "Seeded & Interactive LDA"
* alpha: Hiperparámetro de LDA
* beta: Hiperparámetro de LDA
* topics: Cantidad de tópicos a identificar
* database: Base de datos a utilizar
* nu: Hiperparámetro de LDA cuando se utiliza "Iteractive LDA" o "Seeded & Interactive LDA"
* seeds: Lista de listas, donde cada lista contiene palabras para ser utilizadas en "Seeded LDA", "Interactive LDA" o "Seeded & Interactive LDA".


# .env

Para el correcto funcionamiento de la API, se debe definir el archivo `.env` con las siguientes variables de entorno:

* `token`: llave secreta a utlizar en la API para restringir el acceso
