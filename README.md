# GRL-HumanDiseaseNetwork

Este proyecto implementa métodos de **Aprendizaje de Representación en Grafos (Graph Representation Learning - GRL)** basados en Random Walks (Caminatas Aleatorias) utilizando **Node2Vec** sobre la red de enfermedades humanas (**Human Disease Network**). El objetivo es aprender representaciones vectoriales (embeddings) de las enfermedades y evaluar su calidad mediante tareas posteriores de visualización, clustering y clasificación.

---

## 📂 Estructura del Proyecto

El desarrollo del proyecto está completamente centrado en el notebook principal:

*   **`GRLredes.ipynb`**: Contiene la implementación completa del flujo de trabajo de GRL, clustering y clasificación, así como el análisis detallado y las conclusiones de cada tarea.
*   **`img/`**: Directorio que contiene los gráficos generados durante la ejecución de los experimentos (caminatas aleatorias, proyecciones t-SNE, UMAP y análisis de siluetas).

*(Nota: Los archivos de datos `.csv`, `.edgelist`, `.labels` y los scripts auxiliares `.py` han sido añadidos al `.gitignore` para mantener el repositorio limpio y enfocado exclusivamente en el notebook de trabajo).*

---

## ⚙️ Descripción de la Red de Enfermedades

La red utilizada en este proyecto es un grafo no dirigido y no ponderado en el que:
*   **Nodos**: Representan diferentes enfermedades humanas.
*   **Enlaces**: Conectan dos enfermedades si comparten al menos un gen mutado asociado a ambas patologías.
*   **Datos**: La red consta de **516 nodos** y **1188 enlaces**. Cada enfermedad está etiquetada con su nombre y su categoría fisiológica real (clase de enfermedad), que se utiliza para evaluar los modelos de aprendizaje no supervisado y supervisado.

---

## 🚀 Estado de la Implementación

### TASK 1: Graph Representation Learning (GRL)
Se han generado embeddings de nodo de dimensión 64 explorando diferentes configuraciones de caminatas aleatorias de segundo orden utilizando **Node2Vec**:
1.  **Caminata Aleatoria Clásica (Classic):** Configuración estándar ($p = 1.0, q = 1.0$).
2.  **Sesgo en Profundidad (DFS_bias):** Configuración ($p = 1.0, q = 0.5$) que prioriza la exploración hacia adelante, capturando la **homofilia** (comunidades locales).
3.  **Sesgo en Anchura (BFS_bias):** Configuración ($p = 1.0, q = 2.0$) que restringe la caminata a la vecindad inmediata del nodo inicial, capturando la **equivalencia estructural** (roles en la red).

*   **Visualización:** Se implementó reducción de dimensionalidad a espacio 2D utilizando **t-SNE**, **PCA** y **UMAP** para evaluar visualmente la agrupación de las enfermedades según su clase biológica.

---

### TASK 2: Clustering (Análisis de Agrupamiento)
Se evaluó la capacidad de los embeddings para reconstruir las clases de enfermedades reales utilizando tres algoritmos de clustering con distintas métricas de distancia:
1.  **K-Means** (Distancia Euclídea).
2.  **Clustering Jerárquico (HC)** (Enlace Ward con Distancia Euclídea vs. Enlace Promedio con Similitud del Coseno).
3.  **DBSCAN** (Búsqueda en rejilla de hiperparámetros $\epsilon$ y `min_samples`).

**Conclusiones Clave de la Fase de Clustering:**
*   **Importancia de la métrica de distancia:** En un espacio de 64 dimensiones (curse of dimensionality), la **similitud del Coseno** superó sistemáticamente a la distancia Euclídea. El Clustering Jerárquico con Coseno alcanzó los mejores resultados de NMI (Información Mutua Normalizada).
*   **Calidad de los embeddings:** Las representaciones generadas con **DFS_bias** y **Classic** superaron notablemente a **BFS_bias**. Esto se debe a que las clases biológicas de enfermedad se alinea estrechamente con las comunidades locales del grafo (homofilia) en lugar de los roles estructurales de red (BFS).
*   **Comportamiento de DBSCAN:** Logró identificar de manera efectiva nodos ruidosos (enfermedades huérfanas con baja conectividad), pero debido a la densidad no uniforme del espacio de embeddings, tendió a fragmentar el resto de nodos en micro-clústeres.

---

### TASK 3: Classification (Clasificación Supervisada)
*   **Estado:** *En desarrollo / Próxima fase.*
*   **Objetivo:** Entrenar clasificadores (como `RandomForestClassifier`) utilizando los embeddings de Node2Vec como características (features) para predecir la categoría fisiológica de la enfermedad (`DiseaseClass`).
*   **Enfoque:** Se evaluará el rendimiento del clasificador bajo distintas parametrizaciones del embedding, aplicando métricas rigurosas (Precisión, Recall, F1-Score) y analizando técnicas de selección/reducción de variables.
