# Modelo de demostración de madurez

El checkpoint `modelo_clasificacion_frutas.keras` contiene MobileNetV3Small con pesos iniciales de ImageNet y una cabeza de clasificación entrenada con 2.806 imágenes. Las salidas, en el orden de `clases.json`, son `madura`, `podrida` y `verde`. En VegNet, `podrida` representa la etiqueta `Damaged` de la fuente.

Se usaron 740 imágenes de validación y 1.262 de prueba. Las imágenes de prueba de Kaggle se seleccionaron de su división `test` original. En VegNet se reservaron fechas de captura completas por especie. En total, el subconjunto contiene 4.808 imágenes de manzana, plátano, naranja, tomate y pimiento morrón. [Kaggle](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten) · [VegNet](https://doi.org/10.17632/6nxnjbn9w6.1)

| Prueba | Imágenes | Exactitud | Deterioradas clasificadas como maduras |
| --- | ---: | ---: | ---: |
| General | 1.262 | 95,1 % | 19 de 308 |
| Frutas de Kaggle | 810 | 96,4 % | 7 de 270 |
| Vegetales de VegNet | 452 | 92,7 % | 12 de 38 |

El F1 macro general fue 94,9 %. Los siete tomates deteriorados de prueba se clasificaron como maduros. El pimiento verde quedó fuera de la prueba porque VegNet agrupa esas fotos en una fecha de captura. El prototipo requiere revisión humana y más pruebas externas para usarlo en inspección. [Métricas y matrices de confusión](evaluacion.json)

## Reproducir el entrenamiento

Desde `clasificacion-madurez-frutas`, instala `requirements.txt` y descarga el ZIP de VegNet. La preparación toma el subconjunto de Kaggle v2 mediante solicitudes de rango y registra las particiones en `manifest.csv`.

```sh
python model_training/preparar.py data/subconjunto --vegnet /ruta/vegnet-v1.zip
python model_training/entrenar.py data/subconjunto /ruta/modelo-nuevo
```

Para comparar las métricas, usa TensorFlow 2.21.0, Keras 3.15.1 y la semilla predeterminada. Guarda el `.keras` y su `clases.json` juntos.
