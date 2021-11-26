
En las siguientes notebooks se entrenan las redes neuronales y se guardan los modelos generados en .h5:

- `ANN fs8`
- `ANN Hz`
- `binned_JLA_ANN_DO `
- `binned_JLA_FFNN-v2`
- `binned_JLA_VAE`


Las 4 notebooks que inician con *load*, usan los modelos generados por las redes neuronales, son las ismas notebooks contenidas en el repositorio referido en el artículo del arxiv. 


Las notebooks que inician con  *tunning* usan tensorboard para que, a partir de una malla de hiperparámetros propuesta, se encuentren las mejores combinaciones. En las primeras notebooks mencionadas, se utilizan una de estas combinaciones encontradas.


La notebook `getdist_waCDM_crann` grafica resultados de inferencia de parámetros usando datos provenientes de las redes neuronales y los originales.
