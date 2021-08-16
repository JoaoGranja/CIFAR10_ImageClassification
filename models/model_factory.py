from models.models import resnet101, resnet50, inception, vgg19, vgg16, efficientNet, leNet


def make_model(network, input_shape, num_classes, fine_tuning=False):
    if network == 'LeNet':
        return leNet(input_shape, num_classes, fine_tuning, activation="softmax")
    elif network == 'VGG19':
        return vgg19(input_shape, num_classes, fine_tuning, activation="softmax")
    elif network == 'VGG16':
        return vgg16(input_shape, num_classes, fine_tuning, activation="softmax")
    elif network == 'resnet50':
        return resnet50(input_shape, num_classes, fine_tuning, activation="softmax")
    elif network == 'resnet101':
        return resnet101(input_shape, num_classes, fine_tuning, activation="softmax")
    elif network == 'efficientNet':
        return efficientNet(input_shape, num_classes, fine_tuning, activation="softmax")
    elif network == 'inception':
        return inception(input_shape, num_classes, fine_tuning, activation="softmax")
    else:
        raise ValueError('unknown network ' + network)