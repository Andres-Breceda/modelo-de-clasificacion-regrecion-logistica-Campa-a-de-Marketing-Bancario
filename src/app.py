#El banco portugués está teniendo una disminución en sus ingresos, por lo que quieren poder identificar a los clientes existentes que tienen una mayor probabilidad de contratar un depósito a largo plazo. Esto permitirá que el banco centre sus esfuerzos de marketing en esos clientes y evitará perder dinero y tiempo en clientes que probablemente no se suscribirán.

#Paso 1: Carga del conjunto de datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep=';')

print("cantidad de filas y columnas", data.shape)
print("Nombre de las columnas",data.columns)
data.info()

data.describe()
#Para una distribución aproximadamente normal:
#El 68% de los datos caen dentro de 1 desviación estándar de la media. age singma = 40.2- 10.42, 40.2+ 10.42, 
#El 95% caen dentro de 2 desviaciones estándar.
#El 99.7% caen dentro de 3 desviaciones estándar.

data.head()

#Paso 2: Realiza un EDA completo
#Analisis de variables numericas

# Paso 1: Obtener columnas numéricas
var = data.select_dtypes(include='number').columns.tolist()

# Paso 2: Definir dimensiones fijas del grid
filas = 2
columnas = 5

# Paso 3: Crear subplots
fig, axes = plt.subplots(filas, columnas, figsize=(20, 8))  # Ajusta tamaño como quieras

axes = axes.flatten()  # Aplanar para acceso fácil

# Paso 4: Graficar cada variable
for i, col in enumerate(var):
    if i < len(axes):  # Para no pasarte del número de axes
        sns.histplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(f"Histograma de {col}")

# Si sobran subplots, los apagas
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#Analisis de variables categoricas
#Analisis de variables categoricas
# Visualización de variables categóricas

fig, axis = plt.subplots(2, 6, figsize=(18, 8))  # 1 fila, 3 columnas

sns.histplot(ax=axis[0][0], data=data, x="job")
axis[0][0].set_xticklabels(axis[0][0].get_xticklabels(), rotation=45, ha='right')
sns.histplot(ax=axis[0][1], data=data, x="marital")
axis[0][1].set_xticklabels(axis[0][1].get_xticklabels(), rotation=45, ha='right')
sns.histplot(ax=axis[0][2], data=data, x="education")
axis[0][2].set_xticklabels(axis[0][2].get_xticklabels(), rotation=45, ha='right')
sns.histplot(ax =axis[0][3], data = data, x = "default")
sns.histplot(ax =axis[0][4], data = data, x = "housing")
sns.histplot(ax =axis[0][5], data = data, x = "loan")
sns.histplot(ax =axis[1][0], data = data, x = "contact")
sns.histplot(ax =axis[1][1], data = data, x = "month")
axis[1][1].set_xticklabels(axis[1][1].get_xticklabels(), rotation=45, ha='right')
sns.histplot(ax =axis[1][2], data = data, x = "day_of_week")
sns.histplot(ax =axis[1][3], data = data, x = "poutcome")
sns.histplot(ax =axis[1][4], data = data, x = "y")
axis[1][4].set_title("TARGET")
axis[1][5].remove()
plt.tight_layout()
plt.show()# Guarda imagen


data_original = data.copy()
dc = data.drop(data[data["housing"] == "unknown"].index)
dc = dc.drop(dc[dc["loan"] == "unknown"].index)
dc = dc.drop(dc[dc["marital"] == "unknown"].index)

datafinal = data.drop(data[data["housing"] == "unknown"].index)
datafinalc = datafinal.drop(datafinal[datafinal["loan"] == "unknown"].index)
ddatafinal = datafinal.drop(datafinal[datafinal["marital"] == "unknown"].index)



fig, axis = plt.subplots(2,4 , figsize=(22,20))  # Tamaño más pequeño

sns.countplot(ax=axis[0,0], data=dc, x="job", hue="y")
axis[0,0].set_title("Distribución de trabajos por respuesta")
axis[0,0].set_xlabel("Trabajo")
axis[0,0].set_ylabel("Frecuencia")
axis[0,0].tick_params(axis='x', rotation=30)

sns.countplot(ax=axis[0,1], data=dc, x="marital", hue="y")
axis[0,1].set_title("Distribución de estado civil por respuesta")
axis[0,1].set_xlabel("Estado civil")
axis[0,1].set_ylabel("Frecuencia")
axis[0,1].tick_params(axis='x', rotation=30)

sns.countplot(ax=axis[0,2], data=dc, x="education", hue="y")
axis[0,2].set_title("Distribución por educacion de respuestas")
axis[0,2].set_xlabel("Educacion")
axis[0,2].set_ylabel("Frecuencia")
axis[0,2].tick_params(axis='x', rotation=30)

sns.countplot(ax=axis[0,3], data=dc, x="default", hue="y")
axis[0,3].set_title("Distribución por personas con credito por respuestas")
axis[0,3].set_xlabel("credito")
axis[0,3].set_ylabel("Frecuencia")
axis[0,3].tick_params(axis='x', rotation=30)

sns.countplot(ax=axis[1,0], data=dc, x="housing", hue="y")
axis[1,0].set_title("Personas con prestamo de vivienda por respuestas")
axis[1,0].set_xlabel("Tienen prestamo de vivienda?")
axis[1,0].set_ylabel("Frecuencia")
axis[1,0].tick_params(axis='x', rotation=30)

sns.countplot(ax=axis[1,1], data=dc, x="loan", hue="y")
axis[1,1].set_title("Personas con prestamo de personal por respuestas")
axis[1,1].set_xlabel("Tienen prestamo de personal?")
axis[1,1].set_ylabel("Frecuencia")
axis[1,1].tick_params(axis='x', rotation=30)

sns.countplot(ax=axis[1,2], data=dc, x="poutcome", hue="y")
axis[1,2].set_title("campaña anterior de marketing")
axis[1,2].set_ylabel("Frecuencia")
axis[1,2].tick_params(axis='x', rotation=30)

sns.countplot(ax=axis[1,3], data=dc, x="contact", hue="y")
plt.tight_layout()
plt.show()

#ANALISIS ANOVA

#Hipotesis, "La edad estara relacionada con con las personas que contratan depositos a largo plazo". Podira ser posible debido a que diferentes personas tienen diferentes necesidades financieras.
from scipy.stats import ttest_ind

yes_age = dc[dc['y'] == 'yes']['age']
no_age = dc[dc['y'] == 'no']['age']

t_stat, p_value = ttest_ind(yes_age, no_age)
print(f"T-stat: {t_stat}, P-value: {p_value}")
print("Eso significa que podemos rechazar con confianza la hipótesis nula (H₀) y aceptar la alternativa (H₁). Hay evidencia estadística fuerte de que la edad promedio difiere significativamente entre quienes aceptan y quienes no. Es decir, la edad influye de alguna manera en la decisión.")

sns.histplot(data=dc, x='age', hue='y', kde=True, element='step')
plt.title('Distribución de edades por respuesta')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Seleccionamos solo las columnas numéricas
num_data = datafinal.select_dtypes(include='number')

# Calculamos la matriz de correlación
corr_matrix = num_data.corr()

# Creamos el heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Matriz de correlación entre variables numéricas")
plt.show()

dc["education"]  = pd.factorize(dc["education"])[0]
dc["job"] = pd.factorize(dc["job"])[0]
dc["contact"] = pd.factorize(dc["contact"])[0]
dc["y"] = pd.factorize(dc["y"])[0]
dc["month"] = pd.factorize(dc["month"])[0]

# Ver el resultado

fig, axes = plt.subplots(figsize=(15, 15))

sns.heatmap(dc[["contact", "y", "month", "cons.price.idx", "age", "euribor3m", "nr.employed", "emp.var.rate", "education","job"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

# Draw Plot
plt.show()

#Eliminamos variables que carecen de inportancia en nuestro analisis

# Eliminar columnas solo si existen
columns_to_drop = ["housing", "education","day_of_week", "pdays","previous","loan","default","marital","contact" ]
columns_to_drop = [col for col in columns_to_drop if col in datafinal.columns]  # Filtrar solo las que existen
datafinal= datafinal.drop(columns=columns_to_drop)

print(datafinal.columns)
datafinal

#Descripcion de columnas

# Paso 1: Columnas
columnas = datafinal.columns
n = len(columnas)

# Aquí guardaremos las descripciones como diccionario
descripciones = {}

# Paso 2: Definir grid
filas = (n + 7) // 8  # 8 columnas por fila
fig, axes = plt.subplots(filas, 8, figsize=(20, 3 * filas))
axes = axes.flatten()

# Paso 3: Guardar descripciones y graficarlas
for i, col in enumerate(columnas):
    descripcion = datafinal[col].describe()
    
    # Guardamos en el diccionario
    descripciones[col] = descripcion
    
    # Convertimos a texto para graficarlo
    texto = descripcion.to_string()

    axes[i].axis('off')
    axes[i].text(0, 1, texto, fontsize=9, va='top', ha='left', family='monospace')
    axes[i].set_title(f"Descripción: {col}", fontsize=10, pad=10)

# Apagamos los subplots vacíos
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()


#Limpieza de outlers por columnas

# Crear una copia de `datafinal` para limpiarlo sin modificar el original
datafinal_limpio = datafinal.copy()

# Iteramos sobre las columnas en el diccionario de descripciones
for columna in descripciones:
    descripcion = descripciones[columna]

    # Revisamos que existan los cuartiles
    if "75%" in descripcion.index and "25%" in descripcion.index:
        rango_iqr = descripcion["75%"] - descripcion["25%"]
        limite_superior = descripcion["75%"] + 1.5 * rango_iqr
        limite_inferior = descripcion["25%"] - 1.5 * rango_iqr

        # Poner NaN donde haya outliers
        datafinal_limpio.loc[(datafinal_limpio[columna] < limite_inferior) | (datafinal_limpio[columna] > limite_superior), columna] = np.nan
    else:
        print(f"Saltando columna {columna}: es una variable categórica")

# Eliminar las filas con al menos un NaN (outliers)
datafinal_limpio = datafinal_limpio.dropna()
datosdemodelo= datafinal_limpio
datosdemodelo2= datafinal_limpio
# Mostrar el dataset limpio
datafinal_limpio.head()

print(datafinal.shape, " VS ", datafinal_limpio.shape)

#Paso 3: Construye un modelo de regresión logística

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Definir las características (X) y la variable objetivo (y)
X = datafinal_limpio.drop(columns=['y'])
y = datafinal_limpio['y']

# Si 'y' tiene valores no numéricos, convertirlos usando LabelEncoder
if y.dtypes == 'object':
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

# Crear la columna 'duration_binned' para bins de la duración
bins   = [0, 100, 200, 300, 400, 500, 600, 700]
labels = ['0-99', '100-199', '200-299', '300-399', '400-499', '500-599', '600-699']

X['duration_binned'] = pd.cut(X['duration'], bins=bins, labels=labels, right=False)

# Convertir las variables categóricas en dummies (variables binarias)
X = pd.get_dummies(X, columns=['month','duration_binned','poutcome',"job"])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar solo las características numéricas
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

scaler = StandardScaler()

# Escalar X_train y X_test solo en las columnas numéricas
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# Entrenar el modelo de regresión logística
model = LogisticRegression(C=0.001, penalty=None, solver='sag') #{'C': 0.001, 'penalty': None, 'solver': 'sag'}  C=100, penalty='l2', solver='sag', 
model.fit(X_train, y_train)

#  Aquí cambiamos: obtenemos probabilidades en lugar de predicciones directas
y_prob = model.predict_proba(X_test)[:, 1]

#  Definimos nuestro propio threshold
threshold = 0.3  # Puedes ajustar aquí el número

#  Aplicamos el threshold manualmente
y_pred = np.where(y_prob > threshold, 1, 0)
# Evaluación
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")


# Imprimir los resultados
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

#Paso 4: Optimiza el modelo anterior

from sklearn.model_selection import GridSearchCV

# Definimos los parámetros que queremos ajustar a mano
hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

# Inicializamos la cuadrícula
grid = GridSearchCV(model, hyperparams, scoring = "accuracy", cv = 5)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)

print(f"Mejores hiperparámetros: {grid.best_params_}")

model_grid = LogisticRegression(penalty = "l1", C = 10, solver = "liblinear")
model_grid.fit(X_train, y_train)
y_pred = model_grid.predict(X_test)

grid_accuracy = accuracy_score(y_test, y_pred)
grid_accuracy