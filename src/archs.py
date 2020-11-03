# Network architectures.

from __future__ import division
from utils.ops import relu, conv2d, lrelu, instance_norm, deconv2d

import tensorflow as tf

def discriminator(params, x_input, is_input=False, inps=None):

    """ Discriminator.

    Parameters
    ----------
    params: dict.
    x_init: input tensor.
    reuse: bool, reuse the net if True.

    Returns
    -------
    x_gan: tensor, outputs for adversarial training.
    x_reg: tensor, outputs for gaze estimation.

    """

    layers = 5
    channel = 64
    image_size = params.image_size

    if is_input:
        model = x_input
        x_input = model.get_layer("output").output

    # with tf.variable_scope('discriminator', reuse=reuse):
    # 64 3 -> 32 64 -> 16 128 -> 8 256 -> 4 512 -> 2 1024

    x = conv2d(channel, conv_filters_dim=4, d_h=2, d_w=2, pad=1, use_bias=True)(x_input)
    x = lrelu()(x)

    for i in range(1, layers):
        x = conv2d(channel * 2, conv_filters_dim=4, d_h=2, d_w=2, pad=1, use_bias=True)(x)
        x = lrelu()(x)
        channel = channel * 2

    filter_size = int(image_size / 2 ** layers)

    x_gan = conv2d(1, conv_filters_dim=filter_size, d_h=1, d_w=1, pad=1, use_bias=False)(x)

    x_reg = conv2d(2, conv_filters_dim=filter_size, d_h=1, d_w=1, pad=0, use_bias=False)(x)
    x_reg = tf.reshape(x_reg, [-1, 2])

    if inps is not None:
        model = tf.keras.Model(inputs=inps, outputs=[x_gan, x_reg], name="discriminator")
    else:
        if is_input:
            model = tf.keras.Model(inputs=model.inputs, outputs=[x_gan, x_reg], name="discriminator")
        else:
            model = tf.keras.Model(inputs=x_input, outputs=[x_gan, x_reg], name="discriminator")
    model.summary()
    tf.keras.utils.plot_model(model, "discriminator.png", show_shapes=True)

    return model, x_gan, x_reg
    
def generator(input_, angles, is_input=True, name="gen"):

    """ Generator.

    Parameters
    ----------
    input_: tensor, input images.
    angles: tensor, target gaze direction.
    reuse: bool, reuse the net if True.

    Returns
    -------
    x: tensor, generated image.

    """

    channel = 64
    image_size = 64 ###NOT DYNAMIC
    style_dim = angles.get_shape().as_list()[-1]

    if is_input:
        angles_reshaped = tf.reshape(angles, [-1, 1, 1, style_dim])
        angles_tiled = tf.keras.backend.tile(angles_reshaped, [1, tf.shape(input_)[1], tf.shape(input_)[2], 1])
        x_input = tf.keras.layers.concatenate([input_, angles_tiled], axis=3) 
    else:
        angles_reshaped = tf.reshape(angles, [-1, 1, 1, style_dim])
        angles_tiled = tf.keras.backend.tile(angles_reshaped, [1, tf.shape(input_.get_layer("output").output)[1], tf.shape(input_.get_layer("output").output)[2], 1])
        x_input = tf.keras.layers.concatenate([input_.get_layer("output").output, angles_tiled], axis=3) 

    # angles_reshaped = tf.keras.layers.Reshape([1, 1, style_dim])(angles)
    # angles_tiled = tf.keras.layers.Lambda(tf.keras.backend.tile, arguments={'n':(1, tf.shape(input_)[1], tf.shape(input_)[2], 1)})(angles_reshaped)


    # input layer
    x = conv2d(channel, d_h=1, d_w=1, use_bias=False, pad=3, conv_filters_dim=7)(x_input)
    x = instance_norm()(x)
    x = relu()(x)

    # encoder
    for i in range(2):
        x = conv2d(2 * channel, d_h=2, d_w=2, use_bias=False, pad=1, conv_filters_dim=4)(x)
        x = instance_norm()(x)
        x = relu()(x)
        channel = 2 * channel

    # bottleneck
    for i in range(6):
        x = conv2d(channel, conv_filters_dim=3, d_h=1, d_w=1, pad=1, use_bias=False)(x)
        x = instance_norm()(x)
        x = relu()(x)
        x = conv2d(channel, conv_filters_dim=3, d_h=1, d_w=1, pad=1, use_bias=False)(x)
        x = instance_norm()(x)
        # x = x + x_b

    # decoder
    for i in range(2):
        x = deconv2d(int(channel / 2), conv_filters_dim=4, d_h=2, d_w=2, use_bias=False)(x)
        x = instance_norm()(x)
        x = relu()(x)
        channel = int(channel / 2)


    if is_input:
        x = conv2d(3, conv_filters_dim=7, d_h=1, d_w=1, pad=3, use_bias=False, activation='tanh', name="output")(x)
        model = tf.keras.Model(inputs=[input_, angles], outputs=x, name=name)
    else:
        x = conv2d(3, conv_filters_dim=7, d_h=1, d_w=1, pad=3, use_bias=False, activation='tanh', name="output2")(x)
        model = tf.keras.Model(inputs=[input_.inputs, angles], outputs=x, name=name)

    model.compile()
    model.summary()
    tf.keras.utils.plot_model(model, name+".png", show_shapes=True)
    return model



"""

THE VGG STUFF NEEDS TO BE RE-WRITTEN USING TF2 syntax.

"""
# def vgg_16(inputs, scope='vgg_16', reuse=False):

#     """ VGG-16.

#     Parameters
#     ----------
#     inputs: tensor.
#     scope: name of scope.
#     reuse: reuse the net if True.

#     Returns
#     -------
#     net: tensor, output tensor.
#     end_points: dict, collection of layers.

#     """

#     with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:

#         end_points_collection = sc.original_name_scope + '_end_points'
#         # Collect outputs for conv2d, fully_connected and max_pool2d.
#         with slim.arg_scope(
#                 [slim.conv2d, slim.fully_connected, slim.max_pool2d],
#                 outputs_collections=end_points_collection):
#             net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3],
#                               scope='conv1')
#             net = slim.max_pool2d(net, [2, 2], scope='pool1')
#             net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#             net = slim.max_pool2d(net, [2, 2], scope='pool2')
#             net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#             net = slim.max_pool2d(net, [2, 2], scope='pool3')
#             net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#             net = slim.max_pool2d(net, [2, 2], scope='pool4')
#             net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#             net = slim.max_pool2d(net, [2, 2], scope='pool5')

#             # Convert end_points_collection into a end_point dict.
#             end_points = slim.utils.convert_collection_to_dict(
#                 end_points_collection)

#     return net, end_points
