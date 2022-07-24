﻿cloud_based_potato_disease_classification

A CNN based potato disease classification model is presented in this work. The model is trained in `CNN_based_potato_disease_classification.ipynb' file. The model is served in two ways. One with using fastapi in `api.py' file and second is with using fastapi and tf-serving in `api-tf-serving' file. The PlantVillage dataset obtained from kaggle is also available as a zip file.

The Docker command for creating image of tensorflow model using tf-serving is:

docker run -t --rm -p 8501:8501 -v C:/Users/Muhammad` Shahroz/PycharmProjects/potato-disease:/potato-disease tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-disease/models.config
