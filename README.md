cloud_based_potato_classification

Docker command for tensorflow model serving

docker run -t --rm -p 8501:8501 -v C:/Users/Muhammad` Shahroz/PycharmProjects/potato-disease:/potato-disease tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-disease/models.config
