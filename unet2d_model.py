import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.utils import multi_gpu_model
from GN import GroupNormalization
K.set_image_data_format('channels_last')


# ### Calculating metrics:
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return K.mean((intersection + eps) / (union + eps), axis=0)

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def Jaccard(y_true, y_pred, eps=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + eps) / (union + eps)

def Jaccard_half(y_true, y_pred, eps=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f[y_pred_f < 0.5] = 0
    y_pred_f[y_pred_f >= 0.5] = 1
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + eps) / (union + eps)

def Sensitivity(y_true, y_pred, eps=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + eps) / (K.sum(y_true_f) + eps)

def Sensitivity_half(y_true, y_pred, eps=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f[y_pred_f < 0.5] = 0
    y_pred_f[y_pred_f >= 0.5] = 1
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + eps) / (K.sum(y_true_f) + eps)

def Specificity(y_true, y_pred, eps=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    fp = K.sum(y_pred_f - intersection)
    tn = K.sum((1 - y_pred_f) * (1 - y_true_f))
    return (tn + eps) / (fp + tn + eps)

def Specificity_half(y_true, y_pred, eps=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f[y_pred_f < 0.5] = 0
    y_pred_f[y_pred_f >= 0.5] = 1
    intersection = y_true_f * y_pred_f
    fp = K.sum(y_pred_f - intersection)
    tn = K.sum((1 - y_pred_f) * (1 - y_true_f))
    return (tn + eps) / (fp + tn + eps)

def dice_coefficient(y_true, y_pred, eps=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (K.sum(y_true_f) + K.sum(y_pred_f) + eps)

def dice_coefficient_half(y_true, y_pred, eps=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f[y_pred_f < 0.5] = 0
    y_pred_f[y_pred_f >= 0.5] = 1
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (K.sum(y_true_f) + K.sum(y_pred_f) + eps)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coefficient(y_true, y_pred)



# ### Network Architecture
def downsampling_block_2d(input_tensor, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', batch_normalization=False, activation=None, dropout=None):

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(input_tensor)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    return MaxPooling2D(pool_size=(2, 2))(x), x

def downsampling_gn_block_2d(input_tensor, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', group_normalization=False, activation=None, dropout=None):

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(input_tensor)
    x = GroupNormalization()(x) if group_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = GroupNormalization()(x) if group_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    return MaxPooling2D(pool_size=(2, 2))(x), x


def upsampling_block_2d(input_tensor, skip_tensor, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', batch_normalization=False, activation=None):

    x = UpSampling2D(size=(2, 2))(input_tensor)  #采用上采样
    x = Concatenate()([x, skip_tensor])  # 特征级联

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    return x  #返回第二次卷积的结果

def refined_upsampling_block_2d(input_tensor, skip_tensor, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', batch_normalization=False, activation=None):

    x = UpSampling2D(size=(2, 2))(input_tensor)  #采用上采样
    x = Concatenate()([x, skip_tensor])  # 特征级联

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    x = Conv2D(filters // 2, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

def upsampling_gn_block_2d(input_tensor, skip_tensor, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', group_normalization=False, activation=None):

    x = UpSampling2D(size=(2, 2))(input_tensor)  #采用上采样
    x = Concatenate()([x, skip_tensor])  # 特征级联

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = GroupNormalization()(x) if group_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = GroupNormalization()(x) if group_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    return x  #返回第二次卷积的结果

def deconvsampling_block_2d(input_tensor, skip_tensor, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', batch_normalization=False, activation=None):

    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2))(input_tensor)  #采用反卷积替代上采样
    x = Concatenate()([x, skip_tensor])  # 特征级联

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    return x  #返回第二次卷积的结果

def deconvsampling_gn_block_2d(input_tensor, skip_tensor, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', group_normalization=False, activation=None):

    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2))(input_tensor)  #采用反卷积替代上采样
    x = Concatenate()([x, skip_tensor])  # 特征级联

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = GroupNormalization()(x) if group_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = GroupNormalization()(x) if group_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    return x  #返回第二次卷积的结果

def dense_block_2d(input_tensor, base_filters=32, out_filters=64, block_depth=4, padding='same', activation=ReLU):
    ''' Build a 2d dense_block where the output of each conv_block is fed to subsequent ones
        Args:
            input_tensor: input tensor of dense block.
            base_filters: number of dense block input filters.
            out_filters: number of dense block output filters.
            block_depth: the depths of dense block, make (out_filters - base_filters) / block_depth is a integer.
            padding: conv kernal paddings, default is same
            activation: activation function, default is ReLU
        Returns: keras tensor with out_filters
    '''
    # compute filters of 3x3 conv
    _3x3_filter = int((out_filters - base_filters) / block_depth)

    x = input_tensor
    for i in range(block_depth):
        # temptensor stores the layers which will be concatenate with the output of 3x3 conv
        # x equals the layers which have been concatenated or the input layer
        temptensor = x

        # 1x1 conv + BN + ReLU
        x = Conv2D(base_filters, kernel_size=(1, 1), strides=(1, 1), padding=padding)(x)
        x = BatchNormalization()(x)
        x = activation()(x)

        # 3x3 conv + BN + ReLU
        x = Conv2D(_3x3_filter, kernel_size=(3, 3), strides=(1, 1), padding=padding)(x)
        x = BatchNormalization()(x)
        x = activation()(x)

        # concatenate the tenmtensor with x
        x = Concatenate()([temptensor, x])

    return x

def block_2d(input_tensor, filters, numbersize, kernel_size=(3, 3), strides=(1, 1), padding='same', batch_normalization=True, activation=ReLU):
    _1x1_block = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding)(input_tensor)
    x = input_tensor
    for i in range(numbersize):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x) if batch_normalization else x
        x = activation()(x) if activation else Activation('relu')(x)
    x = Add()([_1x1_block, x])
    return x


# ### Networks
def unet_bn_full_upsampling_dp_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    upsamplingtensors = []  # 用于存放上采样中，第二次卷积的结果，以供之后deep supervision使用
    for i in range(depth):
        x, x0 = downsampling_block_2d(x, n_base_filters, batch_normalization=batch_normalization, activation=activation, dropout=dropout)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    dplayer=None

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = upsampling_block_2d(x, skiptensors[i], n_base_filters, batch_normalization=batch_normalization, activation=activation)
        upsamplingtensors.append(x)
        if i == depth - 1:
            dplayer = upsamplingtensors[depth - i - 1]
        else:
            dplayer = UpSampling2D(size=(2, 2))(dplayer)  # 采用上采样
            dplayer = Concatenate()([upsamplingtensors[depth - i - 1], dplayer])  # 特征级联

    # x = Concatenate()([x, dplayer])
    x = dplayer
    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_bn_block_full_upsampling_dp_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    upsamplingtensors = []  # 用于存放上采样中，第二次卷积的结果，以供之后deep supervision使用
    for i in range(depth):
        x = block_2d(x, n_base_filters, numbersize=min(i+1, depth-1), batch_normalization=batch_normalization, activation=activation)
        skiptensors.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Dropout(0.5)(x) if dropout else x
    x = block_2d(x, n_base_filters, numbersize=depth-1, batch_normalization=batch_normalization, activation=activation)
    dplayer=None

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = UpSampling2D(size=(2, 2))(x)  # 采用上采样
        x = Concatenate()([x, skiptensors[i]])  # 特征级联
        x = block_2d(x, n_base_filters, numbersize=min(i+1, depth-1), batch_normalization=batch_normalization, activation=activation)
        upsamplingtensors.append(x)
        if i == depth - 1:
            dplayer = upsamplingtensors[depth - i - 1]
        else:
            dplayer = UpSampling2D(size=(2, 2))(dplayer)  # 采用上采样
            dplayer = Concatenate()([upsamplingtensors[depth - i - 1], dplayer])  # 特征级联
    # x = Concatenate()([x, dplayer])
    x = dplayer
    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_bn_full_upsampling_dp3_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    upsamplingtensors = []  # 用于存放上采样中，第二次卷积的结果，以供之后deep supervision使用
    for i in range(depth):
        x, x0 = downsampling_block_2d(x, n_base_filters, batch_normalization=batch_normalization, activation=activation,
                                      dropout=dropout)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    dplayer = None

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = upsampling_block_2d(x, skiptensors[i], n_base_filters, batch_normalization=batch_normalization,
                                activation=activation)
        upsamplingtensors.append(x)
        if i == depth - 1:
            dplayer = upsamplingtensors[depth - i - 1]
        else:
            dplayer = UpSampling2D(size=(2, 2))(dplayer)  # 采用上采样
            dplayer = Concatenate()([upsamplingtensors[depth - i - 1], dplayer])  # 特征级联

    # x = Concatenate()([x, dplayer])
    x = dplayer
    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_bn_full_deconv_dp_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    upsamplingtensors = []  # 用于存放上采样中，第二次卷积的结果，以供之后deep supervision使用
    for i in range(depth):
        x, x0 = downsampling_block_2d(x, n_base_filters, batch_normalization=batch_normalization, activation=activation, dropout=dropout)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    dplayer=None

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = deconvsampling_block_2d(x, skiptensors[i], n_base_filters, batch_normalization=batch_normalization, activation=activation)
        upsamplingtensors.append(x)
        if i == depth - 1:
            dplayer = upsamplingtensors[depth - i - 1]
        else:
            dplayer = Conv2DTranspose(n_base_filters, kernel_size=(2, 2), strides=(2, 2))(dplayer)  # 采用反卷积替代上采样
            dplayer = Concatenate()([upsamplingtensors[depth - i - 1], dplayer])  # 特征级联

    # x = Concatenate()([x, dplayer])
    x = dplayer
    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_bn_deconv_upsampling_dp_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    upsamplingtensors = []  # 用于存放上采样中，第二次卷积的结果，以供之后deep supervision使用
    for i in range(depth):
        x, x0 = downsampling_block_2d(x, n_base_filters, batch_normalization=batch_normalization, activation=activation, dropout=dropout)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    dplayer=None

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = deconvsampling_block_2d(x, skiptensors[i], n_base_filters, batch_normalization=batch_normalization, activation=activation)
        upsamplingtensors.append(x)
        if i == depth - 1:
            dplayer = upsamplingtensors[depth - i - 1]
        else:
            dplayer = UpSampling2D(size=(2, 2))(dplayer)  # 采用上采样
            dplayer = Concatenate()([upsamplingtensors[depth - i - 1], dplayer])  # 特征级联

    # x = Concatenate()([x, dplayer])
    x = dplayer
    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_bn_upsampling_deconv_dp_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    upsamplingtensors = []  # 用于存放上采样中，第二次卷积的结果，以供之后deep supervision使用
    for i in range(depth):
        x, x0 = downsampling_block_2d(x, n_base_filters, batch_normalization=batch_normalization, activation=activation, dropout=dropout)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    dplayer=None

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = upsampling_block_2d(x, skiptensors[i], n_base_filters, batch_normalization=batch_normalization, activation=activation)
        upsamplingtensors.append(x)
        if i == depth - 1:
            dplayer = upsamplingtensors[depth - i - 1]
        else:
            dplayer = Conv2DTranspose(n_base_filters, kernel_size=(2, 2), strides=(2, 2))(dplayer)  # 采用反卷积替代上采样
            dplayer = Concatenate()([upsamplingtensors[depth - i - 1], dplayer])  # 特征级联

    # x = Concatenate()([x, dplayer])
    x = dplayer
    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_bn_upsampling_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    for i in range(depth):
        x, x0 = downsampling_block_2d(x, n_base_filters, batch_normalization=batch_normalization, activation=activation, dropout=dropout)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = upsampling_block_2d(x, skiptensors[i], n_base_filters, batch_normalization=batch_normalization, activation=activation)

    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_bn_deconv_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    for i in range(depth):
        x, x0 = downsampling_block_2d(x, n_base_filters, batch_normalization=batch_normalization, activation=activation)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = deconvsampling_block_2d(x, skiptensors[i], n_base_filters, batch_normalization=batch_normalization, activation=activation)

    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_deconv_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def unet_dense_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=32, optimizer=Adam(lr=5e-4), activation=ReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后Concatenate使用

    # 输入层先进行3x3 conv + BN + ReLU
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)

    # 连续depth次的Dense Block + Max Pooling，并存储每次Dense Block的结果用于之后Concatenate使用
    for i in range(depth):
        x = dense_block_2d(input_tensor=x, base_filters=n_base_filters, out_filters=2*n_base_filters)
        skiptensors.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        n_base_filters *= 2

    # 最底层Dense Block卷积操作
    # 此时n_base_filters=512,输出filters为1024
    x = dense_block_2d(input_tensor=x, base_filters=n_base_filters, out_filters=2 * n_base_filters)

    # depth次的上采样,采用stride为(2, 2)的反卷积+BN+ReLU,随后与下采样过程的tensor进行Concatenate,并经过1x1 conv+BN+ReLU后，继续Dense Block
    # i从depth-1到0
    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        # 首先进行stride为(2, 2)的反卷积+BN+ReLU
        x = Conv2DTranspose(n_base_filters, kernel_size=(2, 2), strides=(2, 2))(x)  #采用反卷积替代上采样
        x = BatchNormalization()(x)
        x = activation()(x)

        # 其次进行Concatenate
        x = Concatenate()([skiptensors[i], x])  # 特征级联,i从depth-1到0

        # 随后1x1 conv + BN + ReLU
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = Conv2D(filters=n_base_filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = activation()(x)


        # 送入Dense Block
        if i == 0:
            # i=0即最上方一层时，Dense Block的输入为16，此时n_base_filters为32，需要干预处理
            x = dense_block_2d(input_tensor=x, base_filters=n_base_filters//2, out_filters=2*n_base_filters)
        else:
            x = dense_block_2d(input_tensor=x, base_filters=n_base_filters, out_filters=2 * n_base_filters)

    # 1x1 conv
    x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    outputs = Add()([inputs, x])

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


# GN
def unet_gn_upsampling_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    for i in range(depth):
        x, x0 = downsampling_gn_block_2d(x, n_base_filters, group_normalization=batch_normalization, activation=activation, dropout=dropout)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = GroupNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = GroupNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = upsampling_gn_block_2d(x, skiptensors[i], n_base_filters, group_normalization=batch_normalization, activation=activation)

    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_gn_deconv_2d(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam(lr=5e-4), activation=LeakyReLU, batch_normalization=True, loss_function=dice_coefficient_loss, dropout=None, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    for i in range(depth):
        x, x0 = downsampling_gn_block_2d(x, n_base_filters, group_normalization=batch_normalization, activation=activation)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = GroupNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = GroupNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = deconvsampling_gn_block_2d(x, skiptensors[i], n_base_filters, group_normalization=batch_normalization, activation=activation)

    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', dice_coefficient, Jaccard, Sensitivity, Specificity])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# ### GetNets
def GetNet(model_type='unet_bn_upsampling_2d', net_conf=None):
    if net_conf == None:
        net_conf = {'pretrained_weights': None,
                    'input_size': (256, 256, 1),
                    'depth': 4,
                    'n_base_filters': 64,
                    'optimizer': SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                    'activation': ReLU,
                    'batch_normalization': True,
                    'loss_function': dice_coefficient_loss,
                    'dropout': None,
                    'multi_gpu_num': 0}
    if model_type == 'unet_2d':
        return unet_2d(**net_conf)
    if model_type == 'unet_deconv_2d':
        return unet_deconv_2d(**net_conf)
    if model_type == 'unet_bn_upsampling_2d':
        return unet_bn_upsampling_2d(**net_conf)
    if model_type == 'unet_bn_deconv_2d':
        return unet_bn_deconv_2d(**net_conf)
    if model_type == 'unet_bn_full_upsampling_dp_2d':
        return unet_bn_full_upsampling_dp_2d(**net_conf)
    if model_type == 'unet_bn_full_deconv_dp_2d':
        return unet_bn_full_deconv_dp_2d(**net_conf)
    if model_type == 'unet_bn_deconv_upsampling_dp_2d':
        return unet_bn_deconv_upsampling_dp_2d(**net_conf)
    if model_type == 'unet_bn_upsampling_deconv_dp_2d':
        return unet_bn_upsampling_deconv_dp_2d(**net_conf)
    if model_type == 'unet_dense_2d':
        return unet_dense_2d(**net_conf)
    if model_type == 'unet_bn_block_full_upsampling_dp_2d':
        return unet_bn_block_full_upsampling_dp_2d(**net_conf)
    if model_type == 'unet_gn_upsampling_2d':
        return unet_gn_upsampling_2d(**net_conf)
    if model_type == 'unet_gn_deconv_2d':
        return unet_gn_deconv_2d(**net_conf)


if __name__=='__main__':
    # model = GetNet('unet_2d')
    # model = GetNet('unet_deconv_2d')
    # model = GetNet('unet_bn_upsampling_2d')
    # model = GetNet('unet_bn_deconv_2d')
    # model = GetNet('unet_gn_upsampling_2d')
    # model = GetNet('unet_gn_deconv_2d')
    # model = GetNet('unet_bn_full_upsampling_dp_2d')
    # model = GetNet('unet_bn_full_deconv_dp_2d')
    # model = GetNet('unet_bn_deconv_upsampling_dp_2d')
    # model = GetNet('unet_bn_upsampling_deconv_dp_2d')
    # model = GetNet('unet_dense_2d')
    model = GetNet('unet_bn_deconv_2d')


