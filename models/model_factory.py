from models.models import resnet152_fpn, resnet101_fpn, resnet50_fpn, xception_fpn,  densenet_fpn, LeNet


def make_model(network, input_shape, num_classes):
    if network == 'LeNet':
        return LeNet(input_shape, num_classes, activation="softmax")
    else:
        raise ValueError('unknown network ' + network)