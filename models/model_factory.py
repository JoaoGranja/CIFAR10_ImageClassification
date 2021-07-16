from models.models import resnet152, resnet101, resnet50, xception,  densenet, LeNet


def make_model(network, input_shape, num_classes):
    if network == 'LeNet':
        return LeNet(input_shape, num_classes, activation="softmax")
    elif network == 'resnet152':
        return resnet152(input_shape, num_classes, activation="softmax")
    elif network == 'resnet101':
        return resnet101(input_shape, num_classes, activation="softmax")
    elif network == 'resnet50':
        return resnet50(input_shape, num_classes, activation="softmax")
    elif network == 'resnet18':
        return resnet50(input_shape, num_classes, activation="softmax")
    elif network == 'densenet169':
        return densenet(input_shape, num_classes, activation="softmax")
    elif network == 'xception':
        return xception(input_shape, num_classes, activation="softmax")
    else:
        raise ValueError('unknown network ' + network)