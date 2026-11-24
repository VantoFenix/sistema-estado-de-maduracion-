# Clasificación de madurez de frutas y vegetales

Proyecto de visión por computadora para clasificar imágenes en `madura`, `podrida` y `verde`.

## Arquitectura

- `src/`: configuración compartida, descubrimiento y preparación de datos, CNN Keras, entrenamiento, evaluación, inferencia común y auditoría periódica de drift.
- `api/`: servicio FastAPI, validación de esquemas, predicción, retroalimentación, métricas y registros de monitoreo. `api/static/index.html` contiene la interfaz que también puede servirse desde la API.
- `frontend/`: copia estática reutilizable de la interfaz, lista para desplegar independientemente en Vercel y configurar con la URL del backend.
- `artifacts/`: orden de clases y modelo entrenado. El archivo esperado por la API es `modelo_clasificacion_frutas.keras`.
- `pipelines/`: los trabajos Spark Batch y Kafka/Spark Structured Streaming quedan pendientes para el último bloque.

La API puede arrancar sin el modelo entrenado: `/health` y `/metrics` siguen disponibles, y `/predict` responde `503` hasta que exista el artefacto del modelo.

## Ejecutar localmente

Desde la raíz del proyecto:

```powershell
python -m pip install -r requirements.txt
$env:CORS_ORIGINS = "http://localhost:3000"
uvicorn api.main:app --reload --port 8000
```

La API queda en `http://localhost:8000`; la documentación interactiva está en `/docs`. Para ejecutar la interfaz de Vercel localmente, sirve la carpeta `frontend/` con un servidor de estáticos y configura la URL de API `http://localhost:8000`.

## Despliegue en GitHub, Render y Vercel

### 1. Subir el proyecto a GitHub

Si todavía no existe un repositorio remoto, créalo en GitHub y luego, desde la raíz local del proyecto, ejecuta:

```powershell
git add .
git commit -m "Prepare fruit maturity app for Render and Vercel"
git branch -M main
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main
```

Si `origin` ya existe, omite `git remote add origin`.

### 2. Desplegar la API en Render

1. En Render, selecciona **New + → Web Service** y conecta el repositorio de GitHub.
2. Selecciona la rama `main`, configura el directorio raíz como la raíz del proyecto y el entorno **Docker**.
3. En **Environment**, configura `CORS_ORIGINS` con el origen Vercel, por ejemplo `https://mi-app.vercel.app`. Se permite `*` por defecto para pruebas iniciales; reemplázalo por los dominios reales antes de producción.
4. Crea el servicio y espera al despliegue. Render proporciona `PORT`; el contenedor lo utiliza automáticamente.
5. Verifica `https://<servicio>.onrender.com/health` y `/metrics`.

Para habilitar predicciones, agrega `artifacts/modelo_clasificacion_frutas.keras` al repositorio (el archivo debe respetar el límite de tamaño de GitHub) o configura después un almacenamiento de artefactos accesible durante el despliegue. Sin ese modelo, el servicio inicia correctamente, pero las predicciones devuelven `503`.

Los logs JSON y el baseline se guardan bajo `logs/`. El sistema de archivos de un contenedor puede ser efímero; configura un disco persistente de Render si necesitas conservar esos registros entre despliegues o reinicios.

### 3. Desplegar el frontend en Vercel

1. En Vercel, importa el mismo repositorio de GitHub.
2. Establece **Root Directory** en `frontend` y usa el despliegue estático predeterminado (sin comando de build).
3. Despliega y abre el dominio asignado `https://<proyecto>.vercel.app`.
4. En la sección **Conectar con la API**, ingresa la URL `https://<servicio>.onrender.com` y pulsa **Guardar y probar conexión**. La URL queda guardada en el navegador.

El frontend también acepta `?api=https://<servicio>.onrender.com` o una configuración `window.API_BASE_URL` previa al script. En localhost y al servir desde el host de la API, usa el mismo origen. Asegúrate de que `CORS_ORIGINS` en Render incluya todos los dominios de Vercel usados (producción y, si aplica, preview).

## Pruebas

```powershell
pytest tests -v
```