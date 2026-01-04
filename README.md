# Detección de placas vehiculares
Un detector de placas vehiculares la cual usa paralelismo para la reducción de tiempo.

# Índice

* [Instrucciones de uso](#Instrucciones-de-uso)
* [Herramientas utilizadas](#Herramientas-Utilizadas)
    * [Tesseract OCR](#Tesseract-OCR)
    * [YOLO11](#YOLO11)
    * [RoboFlow](#RoboFlow)
    * [OpenCV](#OpenCV)
* [Funcionamiento](#Funcionamiento)
    * [Procesamiento de imágenes](#Procesamiento-de-imágenes)
    * [Procesamiento de vídeo](#Procesamiento-de-vídeo)
* [Arquitectura del sistema](#Arquitectura-del-sistema)
* [Nota](#Nota)

# Instrucciones de uso

- Clona este repositorio.
- Abre el programa que deseas ejecutar en tu entorno de desarrollo que soporte Python:
    - Las versiones que soporta OpenCV son 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
    - Las versiones que soporta numpy son 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
    - Las versiones que soporta ultralytics son 3.8, 3.9, 3.10, 3.11, 3.12
    - Las versiones que soporta supervision son 3.8, 3.9, 3.10, 3.11, 3.12
    - Las versiones que soporta pytesseract son 3.8, 3.9, 3.10, 3.11, 3.12
- Instalar las siguentes bibliotecas:
```python
# Copiar y pegar lo siguente en el CMD de Windows.
pip install numpy
pip install pytesseract
pip install supervision
pip install ultralytics
pip install opencv-python
```

O tambien usar:
```python
# Copiar y pegar lo siguente en el CMD de Windows.
pip install -r requirements.txt
```

# Herramientas utilizadas

## Tesseract OCR

Es un motor OCR de código abierto que extrae texto impreso o escrito de las imágenes. Fue desarrollado originalmente por Hewlett-Packard, y su desarrollo fue adquirido después por Google. Por eso se conoce ahora como “Google Tesseract OCR”. 

### Instalación

1. Descargar e instalar el ejecutable de [Tesseract OCR](https://github.com/tesseract-ocr/tessdoc?tab=readme-ov-file#binaries)
2. Instalar la biblioteca pytesseract
```python
# Copiar y pegar lo siguente en el CMD de Windows.
pip install pytesseract
```

> [!IMPORTANT]
> Si previamente instalo las bibliotecas de la sección [Instrucciones de uso](#Instrucciones-de-uso), no es necesario volver a instalarlo.

## YOLO11

YOLO11 es la última iteración de la serie Ultralytics YOLO de detectores de objetos en tiempo real, que redefine lo que es posible con una precisión, velocidad y eficacia de vanguardia. Basándose en los impresionantes avances de las versiones anteriores de YOLO , YOLO11 introduce mejoras significativas en la arquitectura y los métodos de entrenamiento, lo que lo convierte en una opción versátil para una amplia gama de tareas de visión por ordenador.

## RoboFlow

Es una plataforma integral que proporciona herramientas para que los desarrolladores construyan, entrenen e implementen modelos de visión por computadora de manera eficiente.

#### [Dataset utilizado de Roboflow Universe Projects](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)

## OpenCV

Es una biblioteca de software de aprendizaje automático y visión artificial de código abierto. OpenCV se creó para proporcionar una infraestructura común para aplicaciones de visión artificial y para acelerar el uso de la percepción artificial en productos comerciales. Al ser un producto con licencia Apache 2, OpenCV facilita a las empresas la utilización y modificación del código.

```python
# Copiar y pegar lo siguente en el CMD de Windows para instalar OpenCV
pip install opencv-python
```

> [!IMPORTANT]
> Si previamente instalo las bibliotecas de la sección [Instrucciones de uso](#Instrucciones-de-uso), no es necesario volver a instalarlo.

# Funcionamiento
## Procesamiento de imágenes

<br> 1. Descargar la base de datos, descomprimir el archivo ZIP y quedarse solo con la carpeta **images**.

> [!IMPORTANT]
> La base de datos de imangenes utilizada para este proyecto pertenece a su resprectivo creador.
> <br><br>Link de la base de datos de las imágenes: https://data.mendeley.com/datasets/nx9xbs4rgx/2

<br> 2. Ejecutar ***change_names.py***.

> [!WARNING]
> Asegurese que la carpeta **images** y el archivo ***change_names.py*** esten en la misma carpeta.

<br> 3. Ejecutar ***recorta.py***.

> [!IMPORTANT]
> Cambiar el valor de *num_threads* a un valor que este dentro del rango de la cantidad de procesadores lógicos de tu procesador

## Procesamiento de vídeo
> [!IMPORTANT]
> Los videos utilizados para este proyecto pertenecen a sus resprectivos creadores.
> <br><br>Link del vídeo *test* y *test2*: https://www.youtube.com/watch?v=QmwIjn6rwQA

<br> 1. Descargar el repositorio o solo el contenido de la carpeta **procesar_videos**.

> [!WARNING]
> Asegurese que se hayan descargado correctamente y que esten en la misma carpeta los videos que el archivo ***detector.py***.

<br> 2. Ejecutar el archivo ***detector.py*** que se encuentra en la carpeta **procesar_videos**.

# Arquitectura del sistema
<image src="diagrama_red_img.svg" alt="Diagrama de red del procesamiento de las imagenes." width="30%" height="30%" />
<image src="diagrama_red_video.svg" alt="Diagrama de red del procesamiento de los videos." width="50%" height="50%" />

# Nota
No se subieron los resultados de los videos con duración de 10 minutos y 1 hora debido a que son pesados. Pero los videos utilizados fueron los siguentes:
> **Duración de 24:43 -** https://www.youtube.com/watch?v=G2v2-H7r6vg
> <br>**Duración de 21:07 -** https://www.youtube.com/watch?v=-UQbT7ncCbs
> <br>**Duración de 3:36:02 -** https://www.youtube.com/watch?v=MTFvfxI3Drk

> [!IMPORTANT]
> Los videos se recortaron para obtener fragmentos de 10 minutos y para 1 hora, se juntaron los videos de duración corta y el largo se obtuvieron 3 fragmentos de 1 hora.

Este proyecto se distribuye bajo la Licencia MIT; puedes usarlo, modificarlo y compartirlo libremente, siempre que se mantenga la atribución correspondiente. Consulta el archivo LICENSE para obtener más detalles.

Si deseas contribuir a este proyecto, puedes enviar solicitudes de extracción (pull requests) con mejoras o características adicionales y si tienes alguna pregunta o problema, puedes contactarme a través de mi perfil de GitHub MrMike92.

2026 | MrMike92 🐢
