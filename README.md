# SDGs-interlinks-with-Climate-Change-Energy-Transition
Conectando los Objetivos de Desarrollo Sostenible con el cambio climático y la transición energética

# 1.- Descripción de proyecto
### Tema
**El cambio climático y la transición energética** hacia una economía descarbonizada es uno de los **grandes desafíos (GD) mundiales** para garantizar que nuestro planeta sobreviva. Para abordar los GD de nuestro planeta, Naciones Unidas ha definido los **Objetivos de Desarrollo Sostenible (ODS)**, un conjunto de **17 objetivos globales**. Los gobiernos del mundo han acordado que estos ODS deben alcanzarse para 2030 *(Guterres, 2019)*.

El conjunto de los 17 ODS se considera uno de los marcos más efectivos para traducir los GD en objetivos que pueden gestionarse para cada país. A pesar de esta visión integral, **existen ciertos condicionantes marcados por un modelo socioeconómico-ambiental y por la situación de los países, que permiten avanzar hacia el logro de algunos ODS sin lograr otros**. Por ello el **objetivo de este proyecto** de Machine Learning es suma importancia, es decir, es imperativo el **estudiar la relación de los ODS con el cambio climático y la transición energética**.

### Datasets y fuentes alternativas de datos
Concentramos el análisis en datos de 137 países a nivel mundial para los años 2017, 2018, 2019 y 2020 sobre los que analizaremos principalmente los 17 ODS en su relación con el cambio climático y la transición energética (9384 observaciones). 

Como datos se usan:
* Los ranking de esos 137 países según los [17 ODS del Informe de Desarrollo Sostenible 2020](https://unstats.un.org/sdgs/dataportal) (Sachs et al., 2020). 
* Según la Agencia Internacional de la Energía, la energía renovable contribuye en un 80% al cambio climático y la transición energética. Por ello, se usa como variable de estudio la inversión en energía renovable (ER) en proyectos totalmente nuevos, y usamos como proxy la generación no hidroeléctrica renovable per cápita en unidades de kWh/cápita. Esta medida se emplea ampliamente en la literatura (Baldwin, Carley, Brass y MacLean, 2016; Carley, 2009; Romano y Scandurra, 2014; Romano, Scandurra, Carfora y Fodor, 2017). Los datos provienen de la base de datos de la [Agencia Internacional de la Energía](https://www.iea.org/data-and-statistics).
* Tres posibles variables cualitativas: país, región y nivel de desarrollo país según lo define la base de datos de lso [Indicadores de Desarrollo Mundial (WDI) del Banco Mundial](https://databank.bancomundial.org/source/world-development-indicators).

### Bibliografía

* *Baldwin, E., Carley, S., Brass, J. N., & MacLean, L. M. 2016. Global renewable electricity policy: A comparative policy analysis of countries by income status. Journal of Comparative Policy Analysis: Research and Practice, 19(3): 277-298.*
* *Carley, S. 2009. State renewable energy electricity policies: An empirical evaluation of effectiveness. Energy Policy, 37(8): 3071-3081.*
* *Guterres, A. 2019. Remarks to high-level political forum on sustainable development. 24 September 2019, United Nations Secretary General.*
* *Romano, A. A., & Scandurra, G. 2014. Investments in renewable energy sources in OPEC members: A dynamic panel approach. Metodoloski Zvezki, 11(2): 93-106.*
* *Romano, A. A., Scandurra, G., Carfora, A., & Fodor, M. 2017. Renewable investments: The impact of green policies in developing and developed countries. Renewable and Sustainable Energy Reviews, 68: 738-747.*
* *Sachs, J., Schmidt-Traub, G., Kroll, C., Lafortune, G., & Fuller, G. 2020. The sustainable development goals and COVID-19. Sustainable development report 2020. Cambridge: Cambridge University Press.*

# 2.- Workflow de Machine Learning
### Definición del Problema
El problema de Machine Learning planteado está alineado con el **objetivo principal** del enunciado del ejercicio: *"crear un modelo predictivo de Machine Learning utilizando los datos conseguidos en la etapa de Data Analysis o explorando nuevos datasets"*.

Dentro del objetivo principal planteado por el enunciado, se plantean otros **objetivos secundarios**:
- Comprensión del **feature importance** de los ODS con respecto al cambio climático y la transición energética
- Comprensión de **Panel Data** en Python: estos modelos que trabajan con data tomados a los mismos n individuos a lo largo de t periodos. Ello nos permite reducir **reducir la colinealidad, la heterogeneidad no observable, y permite reflejar la dinámica y causalidad de Granger"**
- Comprensión del planteamiento utilizando tanto modelos de regresión como clasificación

### Recogida y limpieza de datos
1. **Fuente de datos**: se han cargado los datos de las fuentes indicadas anteriormente.
2. **Outliers**: los outliers no los elimino corresponden principalmente a los países nórdicos y Alemania, potencialmente mas concienciados con el cambio climático y la transición energética.
3. **Missings**: si aplicamos una imputación basada en una regresión lineal, respecto año anterior y posterior.
4. **Anomalías y errores**: se controlan los valores y estan en rangos adecuados.

### EDA
1. **Datos Balanceados**: la variable target en el análisis de clasificación está balanceada. Se dicotomiza la variable target "generación no hidroeléctrica renovable per cápita" con la mediana, para diferenciar los países "wealthy" que tienen mayor inversión en renovables que los países "unwealthy". Nota: en los análisis de regresión se utiliza esta esta variable en su manera contínua.
2. **Correlaciones**: las variables SDG15, SDG14 y SDG17 presentan una correlación < 0.2 con la variables target. Se realizan test para eliminarlas sin mejora significativa, y se decide mantienerlas para ver las interacciones con el resto de los ODSs. Los ODSs con mayor correlacion con la variable target son SDG9 = 0.46, SDG16 = 0.45, y SDG12 = -0.44. No se consideran muy altas.
3. **Plots**: en la clasificación no se detectan efectos significativos entre paises wealthy y unwealthy con las variables dependientes.
4. **Datos**: se recogen datos de Ranking y Scores de los SDGs para los distintos años. Los datos Scores presentan mejores datos y capturan la continuidad de cada uno de los SDGs.
5. **Skewness**: los datos no prentan una Skewness elevada. Se realizan transformaciones logaritmicas, no presentando mejora sustancial.

### ML
1. **Escalados**: se aplica el StandardScaler en los algoritmos de clustering y con los PCAs.
2. **Métricas para clasificación**: se usa el Accuracy en los algoritmos de clasificación, al estar los datos balanceados. Se analiza el Coeficiente de determinacion de la predicción (score) en las predicciones. Se busca la reduccion del Mean Squared Error (MSE) comparar los modelos y poner foco en los errores grandes, no existen muchos outliers (5) que puedan sugerir el uso del Mean Absolute Error (MAE). 
3.- **Regularización**: Se utiliza el GridSearchCV combinada con la técnica de evaluación masiva de parámetros con blucles para facilitar la evaluación de las métricas y regularización.

### MODELOS

Se han utilizado los siguientes modelos:

1.- **Modelos Panel data (regresión)**: Se han utilizado los siguientes modelos:

* Regresión Agrupada (Pooled Regression)
* Efectos fijos (Fixed Effects - FE)
* Efectos aleatorios (Random Effects - RE)

Se violan las condiciones de (Homocedasticidad) y (No autocorrelación) por lo que no se recomienda la Regresión Agrupada. Las condiciones se han probado con una serie de tests diferentes. Para la condición Homocedasticidad, se usa el análisis gráfico. Para la heteroscedasticidad se utiliza el test de White y el test de Breusch-Pagan. Para la condición No autocorrelación, se realiza un test de Durbin-Watson.

El modelo des Efectos fijos es el más adecuados tras realizar el test de Test de Hausman. 

2.- **Modelos PCA y Regresion**: No se presentan mejoras.

3.- **Modelos Pipeline con StandardScaler, PCA y DecissionTree Regression**: proporciona un Coeficiente de determinacion de la predicción en Test de 0.948

4.- **Modelos Pipeline con StandardScaler, PCA y RandomForest Regression**: proporciona un Coeficiente de determinacion de la predicción en Test de 0.9158

5.- **Modelos Pipeline con StandardScaler, PCA y XGBRegressor Regression**: proporciona un Coeficiente de determinacion de la predicción en Test de 0.9143

6.- **Modelo Pipeline con StandardScaler, PCA y KNN Cassification**: proporciona uns Accuracy en Test de 0.8214

Se proporcionan en src el código de los 4 últimos modelos, los restantes se dejan en el directorio src/notebooks. 

En el directorio src/model se han volcado los 4 modelos principales. Para cada uno de ellos se han generado tres ficheros con el mismo formato de los generados en la segunda entrega, es decir: 

        <modelname>: el modelo volcado con pickle
        <modelname>.json: contiene un JSON con la descripción (según la 2da entrega). 
        <modelname>.csv: contiene los datos de test para la variables independientes y dependiente.

En el directorio src/model se deja my_model (y sus 3 ficheros) que es el modelo elegido de **Pipeline con StandardScaler, PCA y DecissionTree Regression**. Se deja un fichero src/train.py que entrena este modelo elegido.

Espero que os guste, ¡gracias!.
