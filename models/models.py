
from keras import Model, Input
from keras.layers import Dense
from tensorflow.keras.utils import get_file

from models.xception_padding import Xception
from models.resnets import ResNet101, ResNet152, ResNet50, ResNet18
#from resnetv2 import InceptionResNetV2Same

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)


def download_resnet_imagenet(v):
    v = int(v.replace('resnet', ''))

    filename = resnet_filename.format(v)
    resource = resnet_resource.format(v)
    if v == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif v == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif v == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'

    return get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )

def resnet152(input_shape, num_classes, activation="softmax"):
    img_input = Input(input_shape)
    resnet_base = ResNet152(img_input, include_top=True)
    resnet_base.load_weights(download_resnet_imagenet("resnet152"))
    x = resnet_base.output
    d1 = Dense(100, activation='relu')(x)
    outputs = Dense(num_classes, activation=activation)(d1)
    model = Model(img_input, x)
    return model


def resnet101(input_shape, num_classes, activation="softmax"):
    img_input = Input(input_shape)
    resnet_base = ResNet101(img_input, include_top=True)
    resnet_base.load_weights(download_resnet_imagenet("resnet101"))
    x = resnet_base.output
    d1 = Dense(100, activation='relu')(x)
    outputs = Dense(num_classes, activation=activation)(d1)
    model = Model(img_input, x)
    return model

def resnet50(input_shape, num_classes, activation="softmax"):
    img_input = Input(input_shape)
    resnet_base = ResNet50(img_input, include_top=True)
    resnet_base.load_weights(download_resnet_imagenet("resnet50"))
    x = resnet_base.output
    d1 = Dense(100, activation='relu')(x)
    outputs = Dense(num_classes, activation=activation)(d1)
    model = Model(img_input, outputs)
    return model

def resnet18(input_shape, num_classes, activation="softmax"):
    img_input = Input(input_shape)
    resnet_base = ResNet18(img_input, include_top=True)
    resnet_base.load_weights(download_resnet_imagenet("resnet18"))
    x = resnet_base.output
    d1 = Dense(100, activation='relu')(x)
    outputs = Dense(num_classes, activation=activation)(d1)
    model = Model(img_input, outputs)
    return model


def xception(input_shape, num_classes, activation="sigmoid"):
    xception = Xception(input_shape=input_shape, include_top=False)
    x = xception.output
    d1 = Dense(100, activation='relu')(x)
    outputs = Dense(num_classes, activation=activation)(d1)
    model = Model(xception.input, x)
    return model


def densenet(input_shape, num_classes, activation="sigmoid"):
    densenet = DenseNet169(input_shape=input_shape, include_top=False)
    x = densenet.output
    d1 = Dense(100, activation='relu')(x)
    outputs = Dense(num_classes, activation=activation)(d1)
    model = Model(densenet.input, x)
    return model

def LeNet(input_shape, num_classes, activation="sigmoid"):   
    
    inputs = Input(input_shape)
    c1 = Conv2D(16, (5, 5), activation='relu', padding='same') (inputs)
    #c1 = Dropout(0.1) (c1)
    c1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (5, 5), activation='relu', padding='same') (c1)
    #c2 = Dropout(0.1) (c2)
    c2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (c2)
    #c3 = Dropout(0.2) (c3)
    c3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (c3)
    #c4 = Dropout(0.2) (c4)
    c4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='relu', padding='same') (c4)
    #c5 = Dropout(0.3) (c5)
    c5 = MaxPooling2D(pool_size=(2, 2)) (c5)
    
    f = Flatten()(c5)

    # Fully Connected 1
    d1 = Dense(1000, activation='relu')(f)
    # Fully Connected 2
    d2 = Dense(100, activation='relu')(d1)
    # Fully Connected 3
    outputs = Dense(num_classes, activation=activation)(d2)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model  


if __name__ == '__main__':
    LeNet((32, 32, 3), 10).summary()

    
    