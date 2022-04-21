# 1.- Descripción de proyecto
### 1.1 Título de proyecto
#### *"Conectando los Objetivos de Desarrollo Sostenible con el cambio climático y la transición energética"*
### 1.2 Tema
**El cambio climático y la transición energética** hacia una economía descarbonizada es uno de los **grandes desafíos (GD) mundiales** para garantizar que nuestro planeta sobreviva. Para abordar los GD de nuestro planeta, Naciones Unidas ha definido los **Objetivos de Desarrollo Sostenible (ODS)**, un conjunto de **17 objetivos globales**. Los gobiernos del mundo han acordado que estos ODS deben alcanzarse para 2030 *(Guterres, 2019)*.
El conjunto de los 17 ODS se considera uno de los marcos más efectivos para traducir los GD en objetivos que pueden gestionarse para cada país. A pesar de esta visión integral, **existen ciertos condicionantes marcados por un modelo socioeconómico-ambiental y por la situación de los países, que permiten avanzar hacia el logro de algunos ODS sin lograr otros**.

### 1.3 Objetivo
 **El objetivo de este proyecto es estudiar la relación de los ODS con el cambio climático y la transición energética**.
 
### 1.4 Datasets y fuentes alternativas de datos
Concentramos el análisis en datos de **137 países** a nivel mundial para los **años 2017, 2018, 2019 y 2020** sobre los que analizaremos principalmente los **17 ODS** en su relación con el **cambio climático y la transición energética (9384 observaciones)**. 
Como datos se usan:
* Los ranking de esos 137 países según los **[17 ODS del Informe de Desarrollo Sostenible 2020](https://unstats.un.org/sdgs/dataportal) (Sachs et al., 2020)**. 
* Según la **Agencia Internacional de la Energía**, la energía renovable contribuye en un 80% al cambio climático y la transición energética. Por ello, se usa como variable de estudio la inversión en energía renovable (ER) en proyectos totalmente nuevos, y usamos como **proxy la generación no hidroeléctrica renovable per cápita en unidades de kWh/cápita**. Esta medida se emplea ampliamente en la literatura (Baldwin, Carley, Brass y MacLean, 2016; Carley, 2009; Romano y Scandurra, 2014; Romano, Scandurra, Carfora y Fodor, 2017). **Los datos provienen de la base de datos de la [Agencia Internacional de la Energía](https://www.iea.org/data-and-statistics)**.
* Tres posibles **variables cualitativas: país, región y nivel de desarrollo país** según lo define la base de datos de los [Indicadores de Desarrollo Mundial (WDI) del Banco Mundial](https://databank.bancomundial.org/source/world-development-indicators).

# 2.- Workflow de Machine Learning
### 2.1 Definición del Problema
Para dar respuesta  **al objetivo de este proyecto es estudiar la relación de los ODS con el cambio climático y la transición energética** se han realizado tres tareas principales, que agrupan tres proyectos en uno, de la sigueiente manera:
 1. **PREDICCIÓN**: que da respuesta a la pregunta... ¿Existen modelos que permitan predecir el “cambio climático y transición energética” a partir de los ODS?
 2. **SEGMENTACIÓN DE LOS ODS**: que da respuesta a la pregunta... ¿ Cuáles son los ODS mas relevantes?
 3. **ANÁLISIS DE LOS COEFICIENTES Y DIRECCIÓN DE LAS VARIABLES DEPENDIENTES**: que da respuesta a la pregunta... ¿Ayuda un ODSs a alcanzar o impedir el “cambio climático y la transición energética” ?
Los modelos utilizados se adecuan a las tareas principales de nuestro desarrollo (predicción, segmentación y análisis de los coeficientes y dirección de las variables dependientes).

Los modelos que se han utilizado para cada una de estas tareas son los siguientes:

#### 2.1.1 Predicción
* Modelos Pipeline PCA RandomForest Regression
* Modelos Pipeline PCA DecissionTree Regression
* Modelo Pipeline PCA XGBRegressor Regression
* Modelo Pipeline PCA KNN Classification

#### 2.1.2 Segmentación
* Modelos KMeans Clustering

#### 2.1.3 Análisis de Coeficientes y dirección
* Modelos Lineal Polinomial Regression 
* Modelos Support Vector Regression 
* Modelos Decission Trees Regression
* Modelos PCAs
* Modelo Panel Data Regression

### 2.2 Recogida y limpieza de datos
1. **Fuente de datos**: se han cargado los datos de las fuentes indicadas anteriormente.
2. **Outliers**: los outliers no se eliminan corresponden principalmente a los países nórdicos y Alemania, potencialmente mas concienciados con el cambio climático y la transición energética.
3. **Missings**: se aplica imputaciones basadas en una regresión lineal, respecto año anterior y posterior.
4. **Anomalías y errores**: se controlan los valores y estan en rangos adecuados.

### 2.3 EDA
Dadas las tareas principales de nuestro desarrollo (predicción, segmentación y análisis de los coeficientes y dirección de las variables dependientes) se utilizarán distintos modelos de regresión, clasificación, reducción de variables y clustering. 
Dado lo extenso del proyecto a continuación de detallan los datos más relevantes encontrados y se proporcionan algunas imagenes de elementos de interés.
1. **Datos Balanceados**: la variable target en el análisis de clasificación está balanceada. Se dicotomiza la variable target "generación no hidroeléctrica renovable per cápita" con la mediana, para diferenciar los países "wealthy" (valor 1 en el target) que tienen mayor inversión en renovables que los países "unwealthy" (valor 0 en el target). Nota: en los análisis de regresión se utiliza esta variable en su manera contínua.
2. **Correlaciones**: las variables SDG15, SDG14 y SDG17 presentan una correlación < 0.2 con la variables target. Se realizan test para eliminarlas sin mejora significativa, y se decide mantienerlas para ver las interacciones con el resto de los ODSs. Los ODSs con mayor correlacion con la variable target son SDG9 = 0.46, SDG16 = 0.45, y SDG12 = -0.44. No se consideran muy altas.
3. **Plots**: en la clasificación se detectan efectos significativos entre paises wealthy vs unwealthy con algunas de las variables dependientes. Las variables SDG1, SDG3, SDG4, SDG7, SDG9 y SDG16 presentan mas paises wealthy para sus valores altos, mientras que SDG13 y SDG3 presentan mas paises wealthy en sus valores bajos.
<p align="center">
<img src="./notebooks/images/SDGs_en_Wealthy_Countries.png" alt="drawing" align="center" width="900"/>
</p>

4. **Datos**: se recogen datos de Ranking y Scores de los SDGs para los distintos años. Los datos Scores presentan mejores datos y capturan la continuidad de cada uno de los SDGs.
5. **Skewness**: los datos no prentan una Skewness elevada. Se realizan transformaciones logaritmicas, no presentando mejora sustancial salvo en la variable y que se utiliza una transformacion logaritmica.
6. **Escalados**: se aplica el StandardScaler en los algoritmos de clustering y con los PCAs.
7. **Métricas para clasificación**: se usa el Accuracy en los algoritmos de clasificación, al estar los datos balanceados. Se analiza el Coeficiente de determinacion de la predicción (score) en las predicciones. Se busca la reduccion del **Mean Squared Error (MSE)** comparar los modelos y poner foco en los errores grandes, no existen muchos outliers (5) que puedan sugerir el uso del Mean Absolute Error (MAE). 
8. **Regularización**: Se utiliza el **GridSearchCV** combinada con la técnica de evaluación masiva de parámetros con blucles para facilitar la evaluación de las métricas y regularización.

#### 2.4.1 REGRESIÓN

**Modelo Pipeline PCA RandomForest Regression**
* Train Split 2021 Coeficiente de determinación: 0.83
* Test Split 2021 Coeficiente de determinación: 0.92
Se les ha pasado en train tan sólo un 80% del año 2021. Generaliza con el resto de los años:
* Test 2017 Coeficiente de determinación: 0.79
* Test 2018 Coeficiente de determinación: 0.76
* Test 2020 Coeficiente de determinación: 0.78

**Modelo Pipeline PCA DecissionTree Regression**
* Train Split 2021 Coeficiente de determinación: 1.0
* Test Split 2021 Coeficiente de determinación: 0.9484216064860829
Se les ha pasado en train tan sólo un 80% del año 2021. Generaliza bien en el 2021, y no adecuadamente con el resto de los años:
* Test 2017 Coeficiente de determinación: 0.16880688640442043
* Test 2018 Coeficiente de determinación: 0.15043052018990943
* Test 2020 Coeficiente de determinación: 0.1906215686369176

**Modelo Pipeline PCA XGBRegressor Regression**
* Train Split 2021 Coeficiente de determinación: 0.98
* Test Split 2021 Coeficiente de determinación: 0.91
Se les ha pasado en train tan sólo un 80% del año 2021. Generaliza bien en el 2021, y no adecuadamente con el resto de los años:
* Test 2017: Coeficiente de determinación : 0.18
* Test 2018: Coeficiente de determinación: 0.16
* Test 2020 Coeficiente de determinación: 0.22


#### 2.4.2 CLASIFICACIÓN

**Modelo Pipeline PCA k-Nearest Neighbors Classification**
* Train Split 2021 Accuracy: 0.88
* Test Split 2021 Accuracy: 0.82
Se les ha pasado en train tan sólo un 80% del año 2021. Generaliza con el resto de los años:
* Test 2017 Accuracy: 0.77
* Test 2018 Accuracy: 0.77
* Test 2020 Accuracy: 0.77

Se proporcionan en src el código de estoss modelos, los restantes se dejan en el directorio src/notebooks. 

En el directorio src/model se han volcado los 4 modelos principales. Para cada uno de ellos se han generado tres ficheros con el mismo formato de los generados en la segunda entrega, es decir: 

        <modelname>: el modelo volcado con pickle
        <modelname>.json: contiene un JSON con la descripción (según la 2da entrega). 
        <modelname>.csv: contiene los datos de test para la variables independientes y dependiente.

En el directorio src/model se deja my_model (y sus 3 ficheros) que es el modelo elegido de **Pipeline con StandardScaler, PCA y DecissionTree Regression**. Se deja un fichero src/train.py que entrena este modelo elegido.

### 2.5 MODELOS USADOS PARA LA SEGMENTACiÓN

**Modelos KMeans Clustering**

Se hace una división de los cuantiles de la puntuación de los paises en cada uno de los SDGS (variables dependientes) y se detecta quien tiene mas paises Wealthy en su 1er (serán los SDGs mas relevantes) y 4to quantile (serán los SDGs menos relevantes). Recordar que los países que están en tienen la **categoria "wealthy" = 1 tienen mayor inversión en renovables (proxy aceptado por la comunidad científicaca del estado de avance del pais en “cambio climático y transición energética”)**. **La relevancia significa que ese SDG en cuestión contribuye mas al avance del país en “cambio climático y transición energética”**.

Los **Clusters 1 y Cluster 0 presentan 43% y 54% de wealthy countries**.
* **Mas relevantes: 1 Quantile SDG1, SDG4, SDG7, SDG13, SDG3**.
* **Menos Relevantes: 4 Quantile SDG14, SDG17, SDG12, SDG15**.

En los **Clusters 1, Cluster 2 y Cluster 2 se detecta**: 
* **Mas relevante: SDG4**
* **Menos relevante: SDG17**

Se pueden observar los detalles en la siguiente figura:
<p align="center">
<img src="./notebooks/images/Modelo_KMeans_Clustering.png" alt="drawing" align="center" width="650"/> 
</p>

### 2.5 MODELOS USADOS PARA EL ANÁLISIS DE COEFICIENTES Y DIRECCIÓN
Se han utilizado los siguientes modelos:

**Modelos Panel data (regresión)**
* **¿PORQUÉ?** : usan algoritmos caja blanca adecuados para n individuos (países), donde los individuos y las variables “x” (ODSs) e “y” (“cambio climático y transición energética”) permanecen iguales a lo largo del tiempo.
* **VENTAJAS**: reduce la colinealidad, captura la heterogeneidad no observable, reflejan la dinámica y causalidad de Granger (causa-efecto), etc.
Dentro de esta categoria se han utilizado los siguientes modelos:
* **Regresión Agrupada (Pooled Regression)**
* **Efectos fijos (Fixed Effects - FE)**
* **Efectos aleatorios (Random Effects - RE)**

En nuestro caso, se violan las condiciones de (**Homocedasticidad**) y (**No autocorrelación**) por lo que no se recomienda la Regresión Agrupada. Las condiciones se han probado con una serie de tests diferentes. Para la condición **Homocedasticidad**, se usa el análisis gráfico. Para la **heteroscedasticidad** se utiliza el **test de White** y el **test de Breusch-Pagan**. Para la condición No autocorrelación, se realiza un **test de Durbin-Watson**.

Finalmente, el modelo des Efectos fijos es el más adecuado tras realizar el test de Test de Hausman. En este caso, se ha transformado logarítmicamente la variable dependiente. Por ello, si el coeficiente de la variable dependiente fuera significativo y tuviera un valor de 0,198, por cada aumento de una unidad en la variable independiente, nuestra variable dependiente aumenta aproximadamente un 21,9% %, calculado como (e^0,198) – 1) * 100 = 21,9%.

Los coeficientes estadísticamente significativos se muestran en la siguiente figura:
<p align="center">
<img src="./notebooks/images/Panel_Data_Coefficientes_Regression.png" alt="drawing" align="center" width="400"/>
</p>

**Modelo “RandomForest Regression y Modelo “XGBRegressor Regression””**

Se ha examinado la feature importance y la permutation importance. Están alineados con los resultados obtenidos en previos métodos

**Modelo “PCAs”**

Se observa como en el PC1 se agrupa SDG12 y SDG13 (0.55 de varianza explicada). Recordemos mantienen una dirección negativa sobre nuestro target.

# 3.- Resultados y conclusiones
## 3.1 TAREA DE PREDICCIÓN
Es factible realizar modelos que predigan “cambio climático y transición energética” a partir de SDG. El reto global del “cambio climático y transición energética” es alcanzable.

## 3.2 AREA SEGMENTACIÓN, ANÁLISIS COEFICIENTES Y DIRECCIÓN
El **SDG4 “Educación de Calidad”** es el Objetivo MAS RELEVANTE 

El **SDG17 “Alianzas para lograr los objetivos”** que consiste en movilizar recursos para desarrollar países subdesarrollados, tiene BAJA RELEVANCIA E INGENUIDAD!. Esto implica:
* **“Gobernanza global imperfecta“**: no existe contrapesos a las desigualdades entre países desarrollados y en desarrollo.
* Una **“trampa de los países no desarrollados”.

En el **SDG9 “Industria, innovación e infraestructura”** se observa:
* Importancia de la **“aglomeración” industrial, clúster tecnológicos,** etc.

En el **SDG12 “Producción y Consumo Responsable”** y **SDG13 “Acción por el Clima”** se oponen al **“cambio climático y transición energética”**. Esto implica:

* La **necesidad de desvincular el SDG 12 y SDG 13 del modelo socioeconómico y ambiental actual** se basa en sistemas heredados de producción y consumo de combustibles fósiles. Ejemplos: la tasa de producción de plástico mundial basada en combustibles fósiles; los petro-estados y subsidios a los combustibles fósiles.
* **“Falsa hipocresía“**: se puede invertir en “transición energética” y estar contaminando con combustibles fósiles mucho mas en el balance neto.

# 3. Bibliografía

* *Baldwin, E., Carley, S., Brass, J. N., & MacLean, L. M. 2016. Global renewable electricity policy: A comparative policy analysis of countries by income status. Journal of Comparative Policy Analysis: Research and Practice, 19(3): 277-298.*
* *Carley, S. 2009. State renewable energy electricity policies: An empirical evaluation of effectiveness. Energy Policy, 37(8): 3071-3081.*
* *Guterres, A. 2019. Remarks to high-level political forum on sustainable development. 24 September 2019, United Nations Secretary General.*
* *Romano, A. A., & Scandurra, G. 2014. Investments in renewable energy sources in OPEC members: A dynamic panel approach. Metodoloski Zvezki, 11(2): 93-106.*
* *Romano, A. A., Scandurra, G., Carfora, A., & Fodor, M. 2017. Renewable investments: The impact of green policies in developing and developed countries. Renewable and Sustainable Energy Reviews, 68: 738-747.*
* *Sachs, J., Schmidt-Traub, G., Kroll, C., Lafortune, G., & Fuller, G. 2020. The sustainable development goals and COVID-19. Sustainable development report 2020. Cambridge: Cambridge University Press.*


Espero que os guste, ¡gracias!.
