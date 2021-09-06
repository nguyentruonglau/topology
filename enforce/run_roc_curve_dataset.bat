::cifar10
python roc_curve_dataset.py --input_dir ./output/data_roc/cifar10/resnet50 --num_class 10
python roc_curve_dataset.py --input_dir ./output/data_roc/cifar10/efficientnetb3 --num_class 10
python roc_curve_dataset.py --input_dir ./output/data_roc/cifar10/inceptionv3 --num_class 10


::fashion mnist
python roc_curve_dataset.py --input_dir ./output/data_roc/fashion_mnist/resnet50 --num_class 10
python roc_curve_dataset.py --input_dir ./output/data_roc/fashion_mnist/efficientnetb3 --num_class 10
python roc_curve_dataset.py --input_dir ./output/data_roc/fashion_mnist/inceptionv3 --num_class 10


::skin canner
python roc_curve_dataset.py --input_dir ./output/data_roc/skin_cancer/resnet50 --num_class 2
python roc_curve_dataset.py --input_dir ./output/data_roc/skin_cancer/efficientnetb3 --num_class 2
python roc_curve_dataset.py --input_dir ./output/data_roc/skin_cancer/inceptionv3 --num_class 2