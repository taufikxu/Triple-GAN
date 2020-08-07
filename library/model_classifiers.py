import torchvision
import Torture
from library import mean_teacher

classifier_dict = {
    "tinyimagenet_cnn": mean_teacher.architectures.tinyimagenet_cnn,
    "cifar10_cnn": mean_teacher.architectures.cifar_cnn,
    "stl10_cnn": mean_teacher.architectures.stl10_cnn,
    "cifar10_cnn_gaussian": mean_teacher.architectures.cifar_cnn_gauss,
    "vat_cnn": mean_teacher.architectures.vat_cnn,
    "cifar10_resnet_26": mean_teacher.architectures.cifar_shakeshake26,
    "cifar10_resnet_pytorch": Torture.Models.Classifier.resnet_cifar.resnet34,
    "stl10_resnet": mean_teacher.architectures.stl10_shakeshake26,
}
