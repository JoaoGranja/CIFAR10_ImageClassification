from models.models import resnet101, resnet50, xception, vgg19, efficientNet, leNet


def make_model(network, input_shape, num_classes):
    if network == 'leNet':
        return leNet(input_shape, num_classes, activation="softmax")
    elif network == 'VGG19':
        return vgg19(input_shape, num_classes, activation="softmax")
    elif network == 'resnet50':
        return resnet50(input_shape, num_classes, activation="softmax")
    elif network == 'resnet101':
        return resnet101(input_shape, num_classes, activation="softmax")
    elif network == 'efficientNet':
        return efficientNet(input_shape, num_classes, activation="softmax")
    elif network == 'xception':
        return xception(input_shape, num_classes, activation="softmax")
    else:
        raise ValueError('unknown network ' + network)