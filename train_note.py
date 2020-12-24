from src.data import Cifar10Data
from src.model import getModel
from src.training import train_model

data = Cifar10Data()

#Give the u_name attribute as anyone of the below mentioned... If u_name is not passed by default it will pick GoogleNet
#GoogleNet #DenseNet # ResNet50
model = getModel(training=True, u_name= ' ', num_classes=data.num_classes)

train_model(model,data.dataloader(),num_epochs=2, save_model_filename="saved_weights_resnet.pt",log_filename="training_logs_resnet.txt"))