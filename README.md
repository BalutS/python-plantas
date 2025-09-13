Funcionamiento Detallado del Programa
1. Ejecución del Programa:
El programa se ejecuta en el entorno Python mediante el archivo App.py. Al 
iniciar, se despliega una interfaz gráfica (GUI) construida con la biblioteca 
Tkinter, la cual permite seleccionar una de las cuatro plantas disponibles 
(Planta 1 a Planta 4) y realizar las operaciones de carga, análisis y 
visualización.
2. Selección de Imágenes:
A través de la interfaz, el usuario carga imágenes una por una desde su 
sistema de archivos. Estas imágenes deben estar organizadas en carpetas 
identificables y, previamente, deben haber sido editadas cuidadosamente
para incluir sobre ellas la imagen de una moneda de 500 pesos 
colombianos (COP). Esta moneda debe estar colocada de manera visible 
sin tapar ninguna parte de la planta. El objetivo de esto es que el programa 
pueda detectar automáticamente la moneda y utilizarla como referencia de 
escala para convertir píxeles en centímetros reales.
3. Medición de la Imagen:
Una vez cargada una imagen, el usuario debe presionar el botón "Medir 
Plantas". En este punto, el programa:
o Detecta las zonas verdes de la imagen para identificar las partes 
correspondientes a la planta.
o Encuentra los contornos y calcula la altura (en píxeles) de cada 
segmento vegetal.
o Detecta la moneda de 500 COP mediante su color característico, 
asumiendo un diámetro estándar de 2.35 cm.
o Calcula la relación entre píxeles y centímetros y convierte la altura de 
la planta a unidades físicas (cm).
o Almacena una copia de la imagen procesada (con contornos y marcas 
de medición) en la carpeta local (imágenes).
4. Almacenamiento de Resultados:
Después de procesar cada imagen, los datos de medición son almacenados 
automáticamente en un archivo CSV (formato Excel) específico para la 
planta seleccionada. Este archivo incluye:
o Fecha y hora del análisis.
o Altura medida.
o Unidad (cm si se detecta la moneda, pixeles si no).
o Ruta de la imagen procesada.
El archivo CSV se guarda en la misma carpeta donde se encuentra el código 
(App.py(mejorado), con nombres como:
growth_data_planta_1.csv, growth_data_planta_2.csv, etc.
5. Análisis Independiente por Planta:
El procedimiento de carga y análisis de imágenes se puede realizar de forma 
independiente para cada una de las cuatro plantas. El usuario puede 
alternar entre las plantas usando el menú desplegable, y cada planta tendrá 
su propio registro y gráfico de crecimiento.
6. Visualización del Crecimiento:
Una vez se han procesado varias imágenes de una misma planta, el usuario 
puede presionar el botón "Ver Crecimiento". El programa generará 
automáticamente:
o Una gráfica temporal con los datos recolectados, mostrando en el eje 
X las fechas de análisis y en el eje Y la altura de la planta en 
centímetros o píxeles.
o Miniaturas de las imágenes procesadas como anotaciones visuales en 
cada punto de la gráfica, lo cual facilita la comprensión del desarrollo 
visual de la planta.
o Esta visualización no solo muestra las tendencias de crecimiento, sino 
que además ayuda a detectar irregularidades o estancamientos en el 
desarrollo.
