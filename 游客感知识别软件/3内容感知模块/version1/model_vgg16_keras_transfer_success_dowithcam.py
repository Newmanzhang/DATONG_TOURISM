
from __future__ import division, print_function
import os
import warnings
import numpy as np

from keras import backend as K
from keras.layers import Input
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers import Conv2D
from keras.regularizers import l2
from keras.layers.core import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

WEIGHTS_PATH = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16_Places365(include_top=True, weights='places',
                    input_tensor=None, input_shape=None,
                    pooling=None,
                    classes=365):
    """Instantiates the VGG16-places365 architecture.
    Optionally loads weights pre-trained
    on Places. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
                 'places' (pre-training on Places),
                 or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape
        """
    if not (weights in {'places', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `places` '
                         '(pre-training on Places), '
                         'or the path to the weights file to be loaded.')

    if weights == 'places' and include_top and classes != 365:
        raise ValueError('If using `weights` as places with `include_top`'
                         ' as true, `classes` should be 365')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv1')(img_input)

    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool", padding='valid')(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv1')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool", padding='valid')(x)

    # Block 3
    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv1')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv2')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool", padding='valid')(x)

    # Block 4
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool", padding='valid')(x)

    # Block 5
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool", padding='valid')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='drop_fc2')(x)

        x = Dense(365, activation='softmax', name="predictions")(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg16-places365')
    # load weights
    if weights == 'places':
        if include_top:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

        model.load_weights(weights_path)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

    elif weights is not None:
        model.load_weights(weights)

    return model


if __name__ == '__main__':
    import urllib
    from keras.applications.vgg16 import (
        VGG16, preprocess_input, decode_predictions)
    from keras.preprocessing import image
    from tensorflow.python.framework import ops
    import tensorflow as tf
    import keras
    import numpy as np
    from PIL import Image
    from cv2 import resize
    import glob
    import cv2
    import matplotlib.pyplot as plt

    # TEST_IMAGE_URL = 'http://places2.csail.mit.edu/imgs/demo/6.jpg'
    model = VGG16_Places365(weights='places')
    # path_test = './Gradcam/'
    # path_file = './didGradcam/'
    # list_name = os.listdir(path_test)
    imgs = []  # 保存测试数据集中的每一张图片
    names = []  # 保存测试数据集中的每一张图片的路径名称
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    '''
    for im in glob.glob(path_test + '/*.jpg'):
        img = cv2.imread(im)
        img = cv2.resize(img, (224, 224))
        imgs.append(img)
        names.append(im)
    imgs = np.asarray(imgs, np.float32)

    image = Image.open('dataset1/val/10839130366_e9e5f5b4d2_c.jpg')
    image = np.array(image, dtype=np.uint8)
    image = resize(image, (224, 224))
    image = np.expand_dims(image, 0)

    prediction = model.predict(imgs)
    datals = []
    #sumnum = {}
    for i in range(np.size(prediction, 0)):
        data = np.around(prediction[i], decimals=4)
        datasort = np.sort(data)[::-1][0:5]
        labelnum = np.argsort(prediction[i])[::-1][0:5]
        sumnum[classes[labelnum[0]]] = sumnum.get(classes[labelnum[0]], 0) + 1
        datajson = {}
        for j in range(np.size(labelnum)):
            datajson[classes[labelnum[j]]] = datasort[j]
        # print(datajson)
        datals.append(datajson)

    #print(sumnum)

    plt.rcParams['figure.figsize']=(12.7,7.2)
    for i in range(np.size(prediction,0)):
    #print("第", i + 1, "个动物:" + flower_dict[prediction[i]])
        img = plt.imread(names[i])
        plt.imshow(img)
        plt.axis('off')
        plt.title(datals[i])
        #plt.show()
        plt.tight_layout()
        plt.savefig("./img_haihe/final{}.jpg".format(i),bbox_inches='tight',transparent=True)
        #plt.show() 

    '''


    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


    def load_image(path):
        img_path = path
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x


    def deprocess_image(x):
        '''
        Same normalization as in:
        https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
        '''
        if np.ndim(x) > 3:
            x = np.squeeze(x)
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_data_format() == "channels_first":
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x


    def register_gradient():
        if "GuidedBackProp" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("GuidedBackProp")
            def _GuidedBackProp(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(grad > 0., dtype) * \
                       tf.cast(op.inputs[0] > 0., dtype)


    def modify_backprop(model, name):
        g = tf.get_default_graph()
        with g.gradient_override_map({'Relu': name}):

            # get layers that have an activation
            layer_dict = [layer for layer in model.layers[1:]
                          if hasattr(layer, 'activation')]

            # replace relu activation
            for layer in layer_dict:
                if layer.activation == keras.activations.relu:
                    layer.activation = tf.nn.relu

            # re-instanciate a new model
            new_model = VGG16_Places365(weights='places')
        return new_model


    def compile_saliency_function(model, activation_layer='block5_pool'):
        input_img = model.input
        layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
        layer_output = layer_dict[activation_layer].output
        max_output = K.max(layer_output, axis=3)
        saliency = K.gradients(K.sum(max_output), input_img)[0]
        return K.function([input_img, K.learning_phase()], [saliency])


    def grad_cam(model, x, category_index, layer_name):
        """
        Args:
           model: model
           x: image input
           category_index: category index
           layer_name: last convolution layer name
        """
        # get category loss
        class_output = model.output[:, category_index]

        # layer output
        convolution_output = model.get_layer(layer_name).output
        # get gradients
        grads = K.gradients(class_output, convolution_output)[0]
        # get convolution output and gradients for input
        gradient_function = K.function([model.input], [convolution_output, grads])

        output, grads_val = gradient_function([x])
        output, grads_val = output[0], grads_val[0]

        # avg
        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)

        # create heat map
        cam = cv2.resize(cam, (x.shape[1], x.shape[2]), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cam)

        # Return to BGR [0..255] from the preprocessed image
        image_rgb = x[0, :]
        image_rgb -= np.min(image_rgb)
        image_rgb = np.minimum(image_rgb, 255)

        cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        cam = np.float32(cam) + np.float32(image_rgb)
        cam = 255 * cam / np.max(cam)
        return np.uint8(cam), heatmap


    pic_folder = "Grad/"
    pic_cam_folder = "./didGradcam/"
    list_name = os.listdir(pic_folder)
    arr_images = []
    for i, file_name in enumerate(list_name):
        img = load_image(pic_folder + file_name)
        predictions = model.predict(img)
        data = np.around(predictions, decimals=4)
        datasort = np.sort(data)[::-1][0]
        top_1 = datasort
        # print('Predicted class:')
        # print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

        predicted_class = np.argmax(predictions)
        cam_image, heat_map = grad_cam(model, img, predicted_class, "block5_pool")

        img_file = image.load_img(pic_folder + list_name[i])
        img_file = image.img_to_array(img_file)

        # guided grad_cam img
        register_gradient()
        guided_model = modify_backprop(model, 'GuidedBackProp')
        saliency_fn = compile_saliency_function(guided_model)

        saliency = saliency_fn([img, 0])
        grad_cam_img = saliency[0] * heat_map[..., np.newaxis]

        # save img
        cam_image = cv2.resize(cam_image, (img_file.shape[1], img_file.shape[0]), cv2.INTER_LINEAR)
        # cv2.putText(cam_image, str(top_1[1]), (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        # cv2.putText(cam_image, str(top_1[2]), (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

        grad_cam_img = deprocess_image(grad_cam_img)
        grad_cam_img = cv2.resize(grad_cam_img, (img_file.shape[1], img_file.shape[0]), cv2.INTER_LINEAR)
        # cv2.putText(grad_cam_img, str(top_1[1]), (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        # cv2.putText(grad_cam_img, str(top_1[2]), (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

        cam_image = cam_image.astype('float32')
        grad_cam_img = grad_cam_img.astype('float32')
        # im_h = cv2.hconcat([img_file, cam_image, grad_cam_img])
        im_h = cv2.hconcat([cam_image, grad_cam_img])
        cv2.imwrite(pic_cam_folder + list_name[i], im_h)
