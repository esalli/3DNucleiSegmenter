# ===============================================
#            Imports
# ===============================================

from tensorflow.keras.layers import Conv2D, Conv3D, Input, Dense, MaxPooling2D, MaxPooling3D, Dropout, UpSampling2D, concatenate, Lambda, UpSampling3D, Add, SpatialDropout3D, SpatialDropout2D, multiply, Activation, Layer, Reshape,  GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow import Tensor


# ===============================================
#            Loss functions and metrics
# ===============================================


def accuracy_metric(num_class=1):
    def accuracy(y, y_pred):
        if num_class == 1:
            y_true = y[:, ..., 0, np.newaxis]
            return tf.keras.metrics.binary_accuracy(y_true, y_pred)
        else:
            y_true = y[:, ..., 0]
            return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

    return accuracy


def tversky_loss(y, y_pred,alpha = 0.5, beta = 0.5):


    '''
    Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
    the score is computed for each class separately and then summed
    alpha=beta=0.5 : dice coefficient
    alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    alpha+beta=1   : produces set of F*-scores
    implemented by E. Moebel, 06/04/18
    '''

    # Separate labels and weight mask
    sample_weight = y[:, ..., 1]
    y_true = y[:, ..., 0]


    y_true = tf.dtypes.cast(y_true, tf.float32) 
    y_pred = tf.dtypes.cast(y_pred, tf.float32) 
    ones = tf.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    axes = tf.range(len(y_true.shape[1:]))+1
    
    num = K.sum(p0*g0, axes)
    den = num + alpha*K.sum(p0*g1,axes) + beta*K.sum(p1*g0,axes)
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    Ncl = K.cast(K.shape(y_true)[-1], 'float32') 
    return Ncl-T


def Tversky(att_weight = 1, alpha = 0.5, beta = 0.5, squared = False):
    def loss(y, y_pred):

        # Separate labels and weight mask
        sample_weight = y[:, ..., 1]
        y_true = y[:, ..., 0]
  
        # Attentive and normal loss
        normal_loss = tversky_loss(y_true, y_pred, alpha, beta, squared)
        att_loss = tversky_loss(sample_weight*y_true, sample_weight*y_pred, alpha, beta, squared)
        return normal_loss + att_weight*att_loss

    return loss


def AM_loss(att_weight = 0, margin = 0.15, scale = 25, batch_size = 12, data_dim = (512, 448, 1)):
    def loss(y, y_pred):

        # Separate labels and weight mask
        sample_weight = y[:, ..., 1]
        y_true = y[:, ..., 0]

        # Additive margin and scaling
        y_pred = y_pred-y_true*margin
        y_pred = y_pred*scale


        if att_weight == 1:
            # Attention
            att_y_true = tf.boolean_mask(y_true, sample_weight)
            att_y_pred = tf.boolean_mask(y_pred, sample_weight)
            loss = K.categorical_crossentropy(att_y_true[:, ..., np.newaxis], att_y_pred, from_logits = True)
        else:
            # Flatten
            size = data_dim
            y_true = tf.reshape(y_true, (size[0]*size[1]*batch_size, 3))
            y_pred = tf.reshape(y_pred, (size[0]*size[1]*batch_size, 3))
            loss = K.categorical_crossentropy(y_true, y_pred, from_logits = True)

        return loss

    return loss



def bin_crossentropy(y, y_pred):

    # Separate labels and weight mask
    sample_weight = y[:, ..., 1]
    y_true = y[:, ..., 0]


    loss = K.binary_crossentropy(y_true[:, ..., np.newaxis], y_pred)
 
    return loss


def dice_loss(y_true, y_pred, squared = True, smooth = 1):

    axes = tf.range(len(y_true.shape[1:]))+1

    intersection = K.sum(y_true * y_pred,axes)
    if squared:
        union = K.sum(y_true, axes) + K.sum(y_pred**2, axes)
    else:
        union = K.sum(y_true, axes) + K.sum(y_pred, axes)
    dice = K.mean( 1-(2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def Dice(att_weight = 1, squared = True):
    def loss(y, y_pred):

        # Separate labels and weight mask
        sample_weight = y[:, ..., 1, np.newaxis]
        y_true = y[:, ..., 0, np.newaxis]

        # Attentive and normal loss
        normal_loss = dice_loss(y_true, y_pred, squared)
        att_loss = dice_loss(sample_weight*y_true, sample_weight*y_pred, squared)
        return normal_loss*(1-att_weight) + att_weight*att_loss
    return loss

def crossent_dice_comb(y, y_pred):

    # Separate labels and weight mask
    sample_weight = y[:, ..., 1]
    y_true = y[:, ..., 0, np.newaxis]

    # Binary crossentropy and dice combined
    loss = K.binary_crossentropy(y_true, y_pred)+dice_loss(y_true, y_pred, squared=True)
 
    return loss




# ===============================================
#            Blocks
# ===============================================

class BinaryThresh(Layer):

    def __init__(self, threshold = 0.5, **kwargs):
        self.threshold = threshold
        super(BinaryThresh, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.threshold
        super(BinaryThresh, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        
        cond = tf.math.less(x, self.threshold*tf.ones(tf.shape(x)))
        out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
        out = tf.concat([x, out], axis = -1)
        return out

    def get_config(self):
        config = super(BinaryThresh, self).get_config()
        config['threshold'] = self.threshold
        return config

def downsample_block(x: Tensor, filters: int, num_conv: int, cbam: int, use_gabor: int, three_dim: int, use_dropout: int): 
    
    for i in np.arange(num_conv):   
        if use_gabor: # Old
                x = Conv2D(kernel_size=3,
                    strides= 1,
                    filters=filters,
                    padding="same", kernel_initializer = 'he_normal', activation = "relu")(x)
        elif three_dim:
                x = Conv3D(kernel_size=3,
                    strides= 1,
                    filters=filters,
                    padding="same", kernel_initializer = 'he_normal', activation = "relu")(x)
        else:   
                x = Conv2D(kernel_size=3,
                    strides= 1,
                    filters=filters,
                    padding="same", kernel_initializer = 'he_normal', activation = "relu")(x)
        if cbam and (i==num_conv-1):
                x = cbam_block_2d(x, ratio = 8)

    x_prev = x

    if use_dropout:
        x = SpatialDropout2D(0.2)(x)

    if three_dim:
        x= MaxPooling3D(pool_size=(2, 2, 1))(x)
    else:
        x = MaxPooling2D(pool_size=(2, 2))(x)
    return x, x_prev

def upsample_block(x: Tensor, x_prev: Tensor, filters: int, num_conv: int, cbam: int, use_gabor: int, three_dim: int, use_dropout: int): 
    
    if three_dim:
        x = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,1))(x))
        x = concatenate([x_prev,x], axis = 4)

    else:
        x = Conv2D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x))
        x = concatenate([x_prev,x], axis = 3)
    for i in np.arange(num_conv):  
        if use_gabor: # Old
                x = Conv2D(kernel_size=3,
                    strides= 1,
                    filters=filters,
                    padding="same", kernel_initializer = 'he_normal', activation = "relu")(x)
        elif three_dim:
                x = Conv3D(kernel_size=3,
                    strides= 1,
                    filters=filters,
                    padding="same", kernel_initializer = 'he_normal', activation = "relu")(x)
        else:           
                x = Conv2D(kernel_size=3,
                    strides= 1,
                    filters=filters,
                    padding="same", kernel_initializer = 'he_normal', activation = "relu")(x)
        if cbam and (i==num_conv-1):
                x = cbam_block_2d(x, ratio = 8)

    if use_dropout:
        x = Dropout(0.2)(x)

    return x

def cbam_block_2d(inputs, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention_2d(inputs, ratio)
    cbam_feature = spatial_attention_2d(cbam_feature)
    return Add()([inputs, cbam_feature])

def channel_attention_2d(input_feature, ratio=8):

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._shape_val[channel_axis]

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)


    return multiply([input_feature, cbam_feature])

def spatial_attention_2d(input_feature):
    kernel_size = 7

    
    channel = input_feature._shape_val[-1]
    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = concatenate([avg_pool, max_pool])
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])


def residual_block(x: Tensor, filters: int): 
    y = Conv2D(kernel_size=3,
               strides= 1,
               filters=filters,
               padding="same", activation = "relu")(x)
    y = Conv2D(kernel_size=3,
               strides=1,
               filters=filters,
               padding="same")(y)

    out = Add()([x, y])
    out = Activation('relu')(out)
    return out

# -----------------------------------------------
#            Unet
# -----------------------------------------------


def unet(input_shape = (256,256,1), num_class = 1, num_filt = 16, num_layers = 5, num_conv = 2, cbam = False, use_gabor = False, three_dim = False, thold = False, use_dropout = False, three_dim_bottom = False):

    # 3D, 2D input shape sanity
    if three_dim and (len(input_shape) < 4):
        input_shape = input_shape + (1,)

    if not three_dim and (len(input_shape) > 3):
        input_shape = input_shape[0:3]

    inputs = Input(input_shape)
    x = inputs

    # Thresholding
    if thold:
        x = BinaryThresh(threshold = 0.7)(x)

    # Encoder
    x_prevs = []
    for i in np.arange(num_layers):
        x, x_prev = downsample_block(x, num_filt*(i+1), num_conv, cbam, use_gabor, three_dim, use_dropout)
        x_prevs.append(x_prev)
        
    _, x_prev = downsample_block(x, num_filt*(i+1), num_conv, cbam, use_gabor, three_dim, use_dropout)
    x=x_prev

    # Decoder
    for i in np.arange(num_layers):
        x= upsample_block(x, x_prevs[num_layers-1-i], num_filt*(num_layers-i), num_conv, cbam, use_gabor, three_dim, use_dropout)

    # Output
    if num_class == 1:
        output = Dense(units = 1, activation = 'sigmoid')(x)

    else:
        output = Dense(units= num_class, activation = 'softmax')(x)


    if three_dim and three_dim_bottom:
        output = Lambda(lambda x: K.mean(x, axis=3))(output)
        output = Dense(units= num_class, activation = 'sigmoid')(output)

    model = Model(inputs = inputs, outputs = output)
    return model

# ===============================================
#            Main
# ===============================================

'''
Model and loss function configurations. Slightly more than what investigated in the paper
'''

if __name__ == '__main__':
    print("Nothing to run here.")
