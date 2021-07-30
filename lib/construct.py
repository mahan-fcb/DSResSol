
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Activation, Reshape, Multiply, Add
from tensorflow.keras.layers import Conv1D, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import concatenate


def synthesis_block(latents, image, filters, num_filters=2, kernal_size=3, axis=-1, strides=1, alpha=0.2):
    g = Conv1D(num_filters*filters, kernel_size=kernal_size,
               padding='same')(latents)
    g = BatchNormalization(axis=axis)(g)
    g = LeakyReLU(alpha=alpha)(g)
    g = Conv1D(num_filters*filters, kernel_size=kernal_size,
               strides=strides, padding='same')(g)
    g = BatchNormalization(axis=axis)(g)
    g_lat = g[:, :, :filters]
    g_switch = Activation('sigmoid')(g[:, :, filters:])
    x = Multiply()([g_switch, image])
    g = Add()([x, g_lat])
    return g


def SE_block(input_shape):
    x = GlobalMaxPooling1D()(input_shape)
    x = Dense(int(int(input_shape.shape[-1])/8), activation='relu')(x)
    x = Dense(int(input_shape.shape[-1]), activation='sigmoid')(x)
    x = Reshape((-1, int(input_shape.shape[-1])))(x)
    x = Multiply()([x, input_shape])
    return x


def residual_block(data, filters, filter_size, d_rate, leaning_rate=0.001):
    data = Conv1D(filters, filter_size, padding='same',
                  kernel_regularizer=l2(leaning_rate))(data)
    data = Conv1D(filters, 3, dilation_rate=d_rate, padding='same',
                  kernel_regularizer=l2(leaning_rate))(data)
    bn1 = BatchNormalization()(data)
    act1 = Activation('relu')(bn1)
    conv1 = Conv1D(filters, 1, padding='same',
                   kernel_regularizer=l2(leaning_rate))(act1)
    conv = SE_block(conv1)
    x = concatenate([conv, data])
    return x
