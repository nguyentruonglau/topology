:: SKIN CANCER DATASET

:: RESNET50
python main.py --model_name ResNet50 --data_shape 192 256 3 --model_shape 224 224 --data_name skin_cancer --class_name melanoma --num_img_train 3600 --num_img_test 450
python main.py --model_name ResNet50 --data_shape 192 256 3 --model_shape 224 224 --data_name skin_cancer --class_name nevus --num_img_train 3600 --num_img_test 450


:: EFFICIENTNETB3
python main.py --model_name EfficientNetB3 --data_shape 192 256 3 --model_shape 300 300 --data_name skin_cancer --class_name melanoma --num_img_train 3600 --num_img_test 450
python main.py --model_name EfficientNetB3 --data_shape 192 256 3 --model_shape 300 300 --data_name skin_cancer --class_name nevus --num_img_train 3600 --num_img_test 450


:: INCEPTIONV3
python main.py --model_name InceptionV3 --data_shape 192 256 3 --model_shape 299 299 --data_name skin_cancer --class_name melanoma --num_img_train 3600 --num_img_test 450
python main.py --model_name InceptionV3 --data_shape 192 256 3 --model_shape 299 299 --data_name skin_cancer --class_name nevus --num_img_train 3600 --num_img_test 450