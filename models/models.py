
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet101, EfficientNetB0, InceptionV3


def vgg16(input_shape, num_classes, fine_tuning=False, activation="softmax"):
    img_input = Input(shape=input_shape)
    base_model = VGG16(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape, 
    )  # Do not include the ImageNet classifier at the top.
    # Freeze the base_model
    base_model.trainable = fine_tuning
    
    #Build the model
    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="VGG16")
    return model

def vgg19(input_shape, num_classes, fine_tuning=False, activation="softmax"):
    img_input = Input(shape=input_shape)
    base_model = VGG19(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape, 
    )  # Do not include the ImageNet classifier at the top.
    # Freeze the base_model
    base_model.trainable = fine_tuning
    
    #Build the model
    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="VGG19")  
    return model

def resnet50(input_shape, num_classes, fine_tuning=False, activation="softmax"):
    img_input = Input(shape=input_shape)
    base_model = ResNet50(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape,  
                    classes=10,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = fine_tuning
    
    #Build the model
    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="resnet50")  
    return model

def resnet101(input_shape, num_classes, fine_tuning=False, activation="softmax"):
    img_input = Input(shape=input_shape)
    base_model = ResNet101(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape,  
                    classes=10,
    )  # Do not include the ImageNet classifier at the top.
    # Freeze the base_model
    base_model.trainable = fine_tuning
    
    #Build the model
    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="resnet101")  
    return model

def inception(input_shape, num_classes, fine_tuning=False, activation="softmax"):
    img_input = Input(shape=input_shape)
    base_model = InceptionV3(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape,  
                    classes=10,
    )  # Do not include the ImageNet classifier at the top.
    # Freeze the base_model
    base_model.trainable = fine_tuning
    
    #Build the model
    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="inception")  
    return model

def efficientNet(input_shape, num_classes, fine_tuning=False, activation="softmax"):
    img_input = Input(shape=input_shape)
    base_model = EfficientNetB0(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape,  
                    classes=10,
    )  # Do not include the ImageNet classifier at the top.
    # Freeze the base_model
    base_model.trainable = fine_tuning
    
    #Build the model
    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="efficientNet")  
    return model

def leNet(input_shape, num_classes, fine_tuning=False, activation="softmax"):   
    
    inputs = Input(input_shape)
    c1 = Conv2D(16, (5, 5), activation='relu', padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1) (c1)
    c1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (5, 5), activation='relu', padding='same') (c1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1) (c2)
    c2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (c2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2) (c3)
    c3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (c3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2) (c4)
    c4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='relu', padding='same') (c4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3) (c5)
    c5 = MaxPooling2D(pool_size=(2, 2)) (c5)
    
    f = Flatten()(c5)

    # Fully Connected 1
    d1 = Dense(1000, activation='relu')(f)
    d1 = BatchNormalization()(d1)
    d1 = Dropout(0.3) (d1)
    # Fully Connected 2
    d2 = Dense(100, activation='relu')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Dropout(0.3) (d2)
    # Fully Connected 3
    outputs = Dense(num_classes, activation=activation)(d2)
    
    model = Model(inputs=[inputs], outputs=[outputs], name="LeNet")
    return model  


if __name__ == '__main__':
    leNet((32, 32, 3), 10).summary()

    
    