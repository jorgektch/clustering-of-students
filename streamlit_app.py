import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Análisis de Clustering en Perfiles de Redes Sociales de Estudiantes", layout="wide")
st.title("Análisis de Agrupamiento en Perfiles de Redes Sociales de Estudiantes")

# Descripción del estudio
st.markdown("""
Este estudio presenta un análisis de agrupamiento aplicado a un dataset de perfiles de redes sociales de estudiantes universitarios. 
El objetivo es identificar patrones de comportamiento en redes sociales segmentando los datos en grupos homogéneos.
""")

# Cargar datos
@st.cache_data
def load_data():
    url = "https://www.kaggle.com/datasets/zabihullah18/students-social-network-profile-clustering/download"
    data = pd.read_csv(url)
    return data

data = load_data()
st.write("## Datos Iniciales")
st.write(data.head())

# Preprocesamiento
st.write("## Preprocesamiento")
data = data.dropna()
st.write("Datos después de eliminar valores nulos:", data.shape)

# Codificación y estandarización
data = pd.get_dummies(data, columns=['gradyear', 'gender'])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Gráfico PCA
st.write("## Reducción de Dimensionalidad con PCA")
fig, ax = plt.subplots()
ax.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
ax.set_title("Visualización de Datos en las dos Primeras Componentes Principales")
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
st.pyplot(fig)

# Determinación del número de clústeres
st.write("## Determinación del Número de Clústeres")
st.write("### Método del Codo")
inertia = []
range_clusters = range(1, 11)
for n_clusters in range_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_pca)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range_clusters, inertia, marker='o')
ax.set_title("Método del Codo")
ax.set_xlabel("Número de Clústeres")
ax.set_ylabel("Inercia")
st.pyplot(fig)

# Coeficiente de Silueta
st.write("### Coeficiente de Silueta")
silhouette_scores = []
for n_clusters in range_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_pca)
    silhouette_avg = silhouette_score(data_pca, labels)
    silhouette_scores.append(silhouette_avg)

fig, ax = plt.subplots()
ax.plot(range_clusters, silhouette_scores, marker='o')
ax.set_title("Coeficiente de Silueta")
ax.set_xlabel("Número de Clústeres")
ax.set_ylabel("Puntaje Silueta")
st.pyplot(fig)

# Selección del número de clústeres
n_clusters = st.slider("Seleccione el Número de Clústeres:", min_value=2, max_value=10, value=2)

# Aplicación de K-Means con el número de clústeres seleccionado
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_pca)

# Validación cruzada y resultados
st.write("## Resultados de Clustering")
st.write("Número de Clústeres Seleccionado:", n_clusters)
scores = cross_val_score(kmeans, data_pca, data['Cluster'], cv=5)
st.write("Puntajes de Validación Cruzada:", scores)
st.write("Puntaje promedio de CV:", np.mean(scores))

# Conteo de Clústeres
cluster_counts = data['Cluster'].value_counts()
st.write("### Conteo de Clústeres")
st.write(cluster_counts)

# Aplicación de SMOTE
smote = SMOTE()
data_resampled, labels_resampled = smote.fit_resample(data.drop(columns=['Cluster']), data['Cluster'])
st.write("Datos después de SMOTE:", data_resampled.shape, labels_resampled.shape)

# Resumen estadístico por clúster
st.write("### Resumen Estadístico por Clúster")
cluster_summary = data.groupby('Cluster').mean()
st.write(cluster_summary)

# Visualización de los clústeres en el espacio PCA
fig, ax = plt.subplots()
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['Cluster'], palette="viridis", ax=ax)
ax.set_title("Visualización de Clústeres en el Espacio PCA")
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
st.pyplot(fig)

# Conclusiones
st.write("## Conclusiones")
st.markdown("""
El análisis de agrupamiento aplicado a los perfiles de redes sociales de estudiantes permitió observar cómo ciertos perfiles tienden a agruparse en función de variables comunes, como deportes y preferencias de marcas.
Estos resultados abren posibilidades para el diseño de campañas personalizadas y estrategias de marketing enfocadas a segmentos específicos.
""")
