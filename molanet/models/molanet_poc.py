import tensorflow as tf
from PIL import Image
import numpy as np


def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02))
    # return tf.get_variable(name, shape, initializer=tf.uniform_unit_scaling_initializer(factor=2.0))


def bias_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def leaky_relu(features, alpha=0.0):
    return tf.maximum(alpha * features, features)


def conv2d(features, feature_count, name, use_batchnorm=True, strides=None, filter_sizes=None, do_activation=True, padding="SAME", input_channels=None):
    if filter_sizes is None:
        filter_sizes = [4, 4]
    if strides is None:
        strides = [2, 2]
    if input_channels is None:
        input_channels = features.get_shape()[-1]

    w = weight_variable("w_" + name, [filter_sizes[0], filter_sizes[1], input_channels, feature_count])
    b = bias_variable("b_" + name, [feature_count])
    conv = tf.nn.bias_add(tf.nn.conv2d(features, w, strides=[1, strides[0], strides[1], 1], padding=padding), b)
    if use_batchnorm:
        bn = tf.contrib.layers.batch_norm(conv, decay=0.9, epsilon=1e-5)  # TODO: Params?
    else:
        bn = conv

    if do_activation:
        a = leaky_relu(bn, 0.2)
    else:
        a = bn

    return a, w, b


def conv2d_transpose(features, feature_count, output_size, name, keep_prob, batch_size, strides=None, filter_sizes=None, concat_activations=None, use_batchnorm=True, do_activation=True, padding="SAME"):
    if filter_sizes is None:
        filter_sizes = [4, 4]
    if strides is None:
        strides = [2, 2]

    w = weight_variable("w_" + name, [filter_sizes[0], filter_sizes[1], feature_count, features.get_shape()[-1]])
    b = bias_variable("b_" + name, [feature_count])
    conv = tf.nn.bias_add(tf.nn.conv2d_transpose(features, w, output_shape=[batch_size, output_size[0], output_size[1], feature_count], strides=[1, strides[0], strides[1], 1], padding=padding), b)
    if use_batchnorm:
        bn = tf.contrib.layers.batch_norm(conv, decay=0.9, epsilon=1e-5)  # TODO: Params?
    else:
        bn = conv
    d = tf.nn.dropout(bn, keep_prob)

    if do_activation:
        a = tf.nn.relu(d)
    else:
        a = d

    # Concat activations if available
    if concat_activations is not None:
        a = tf.concat([a, concat_activations], axis=3, name=f"{name}_skipconn")

    return a, w, b




class MolanetPoc(object):

    @staticmethod
    def _create_generator(image: tf.Tensor, keep_prob: tf.Tensor=tf.constant(0.5), reuse=False, reshape=False):
        with tf.variable_scope("generator", reuse=reuse):
            # Patch size: 70

            # input_tensor = tf.extract_image_patches(image, [1, 94, 94, 1], [1, 70, 70, 1], rates=[1, 1, 1, 1], padding="VALID")
            # input_tensor = tf.reshape(input_tensor, [-1, 94, 94, 3])

            input_tensor = image

            # batch_size = patches_y * patches_x
            batch_size = tf.shape(input_tensor)[0]

            a_enc1, _, _ = conv2d(input_tensor, 64, "enc_1", use_batchnorm=False, padding="VALID", input_channels=3)
            a_enc2, _, _ = conv2d(a_enc1, 128, "enc_2", padding="VALID")
            a_enc3, _, _ = conv2d(a_enc2, 256, "enc_3", padding="VALID")
            a_enc4, _, _ = conv2d(a_enc3, 256, "enc_4", padding="VALID")
            a_enc5, _, _ = conv2d(a_enc4, 256, "enc_5", padding="VALID")

            a_dec1, _, _ = conv2d_transpose(a_enc5, 256, [4, 4], "dec_1", keep_prob, batch_size, concat_activations=a_enc4, use_batchnorm=False, padding="VALID")
            a_dec2, _, _ = conv2d_transpose(a_dec1, 256, [10, 10], "dec_2", keep_prob, batch_size, concat_activations=a_enc3, padding="VALID")
            a_dec3, _, _ = conv2d_transpose(a_dec2, 128, [22, 22], "dec_3", keep_prob, batch_size, concat_activations=a_enc2, padding="VALID")
            a_dec4, _, _ = conv2d_transpose(a_dec3, 64, [46, 46], "dec_4", 1.0, batch_size, concat_activations=a_enc1, padding="VALID")
            a_dec5, _, _ = conv2d_transpose(a_dec4, 1, [94, 94], "dec_5", 1.0, batch_size, concat_activations=input_tensor, padding="VALID")

            logits, _, _ = conv2d(a_dec5, 1, "dec_out", strides=[1, 1], filter_sizes=[25, 25], use_batchnorm=False, do_activation=False, padding="VALID", input_channels=4)

            # TODO: Might not work
            if reshape:
                logits = tf.reshape(logits, [1, 11 * 70, 15 * 70, 1]) # TODO

            a_out = tf.tanh(logits, "a_out")

            return a_out, logits


    @staticmethod
    def _create_discriminator(image: tf.Tensor, mask: tf.Tensor, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):

            # Patch size: 70
            input_tensor = tf.concat([image, mask], axis=3)

            # 70x70x4 -> 64x64x64
            a1, w1, b1 = conv2d(input_tensor, 64, "disc_1", strides=[1, 1], filter_sizes=[7, 7], use_batchnorm=False, padding="VALID", input_channels=4)

            # 64x64x64 -> 32x32x128
            a2, w2, b2 = conv2d(a1, 128, "disc_2")

            # 32x32x128 -> 16x16x256
            a3, w3, b3 = conv2d(a2, 256, "disc_3")

            # 16x16x256 -> 8x8x512
            a4, w4, b4 = conv2d(a3, 512, "disc_4")

            # 8x8x512 -> 4x4x512
            a5, w5, b5 = conv2d(a4, 512, "disc_5")

            # 4x4x512 -> 2x2x512
            a6, w6, b6 = conv2d(a5, 512, "disc_6")

            # 2x2x512 -> 1x1x1
            logits, w_out, b_out = conv2d(a6, 1, "disc_out", use_batchnorm=False, do_activation=False)

            # Reshape logits back
            # TODO: Can't do this anymore right now
            #logits = tf.reshape(logits, [1, patches_y, patches_x, 1])

            # Actual value
            a_out = tf.reduce_mean(logits)
            return a_out, logits


    @staticmethod
    def _pad_image(image: tf.Tensor, network_shape: tf.TensorShape, output_shape: tf.TensorShape) -> tf.Tensor:

        output_height = output_shape.as_list()[1]
        output_width = output_shape.as_list()[2]
        network_height = network_shape.as_list()[1]
        network_width = network_shape.as_list()[2]
        image_height = tf.shape(image)[1]
        image_width = tf.shape(image)[2]

        network_hotspot = tf.cast(tf.floor((network_height - 1) / 2), dtype=tf.int32)
        output_hotspot = tf.cast(tf.floor((output_height - 1) / 2), dtype=tf.int32)
        padding_left_top = network_hotspot - output_hotspot

        # remainder = image_size % output_size
        # # (OW - R) % OW + IW - HSI - (OW - HS0)
        # return (output_size - remainder) % output_size + \
        #        filter_size - calculate_hotspot(filter_size) - (output_size - calculate_hotspot(output_size))

        remainder_bottom = tf.mod(image_height, output_height)
        padding_bottom = tf.mod((output_height - remainder_bottom), output_height) + \
             network_height - network_hotspot - (output_height - output_hotspot)

        remainder_right = tf.mod(image_width, output_width)
        padding_right = tf.mod((output_width - remainder_right), output_width) + \
            network_width - network_hotspot - (output_width - output_hotspot)

        return tf.pad(image, [[0, 0], [padding_left_top, padding_bottom], [padding_left_top, padding_right], [0, 0]], mode="SYMMETRIC")

    @staticmethod
    def split_discriminator_image(input_tensor):
        pad_bottom = tf.mod(70 - tf.mod(tf.shape(input_tensor)[1], 70), 70)
        pad_right = tf.mod(70 - tf.mod(tf.shape(input_tensor)[2], 70), 70)
        input_tensor = tf.pad(input_tensor, [[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]], mode="CONSTANT")

        # patches_y = tf.cast(tf.divide(tf.shape(input_tensor)[1], 70), tf.int32)
        # patches_x = tf.cast(tf.divide(tf.shape(input_tensor)[2], 70), tf.int32)

        channel_count = tf.shape(input_tensor)[3]

        input_tensor = tf.extract_image_patches(input_tensor, [1, 70, 70, 1], [1, 70, 70, 1], rates=[1, 1, 1, 1], padding="VALID")
        input_tensor = tf.reshape(input_tensor, [-1, 70, 70, channel_count])
        return input_tensor

    def __init__(self):
      pass

if __name__ == "__main__":

    in_image = np.asarray(Image.open("/tmp/image.jpg"), dtype=np.float32)
    in_image = (in_image / 255.0 - 0.5) * 2.0
    in_image = np.reshape(in_image, (1, in_image.shape[0], in_image.shape[1], in_image.shape[2]))

    in_segmentation = np.asarray(Image.open("/tmp/segmentation.png"), dtype=np.float32)
    in_segmentation = (in_segmentation / 255.0 - 0.5) * 2.0
    in_segmentation = np.reshape(in_segmentation, (1, in_segmentation.shape[0], in_segmentation.shape[1], 1))

    # image = tf.placeholder(tf.float32, [1, None, None, 3], name="image")
    image = tf.placeholder(tf.float32, [None, 94, 94, 3], name="image")
    image_patches = tf.placeholder(tf.float32, [None, 70, 70, 3], name="image_patches")
    mask_patches = tf.placeholder(tf.float32, [None, 70, 70, 1], name="mask_patches")

    # padded_image = tf.reshape(padded_image, [1, 794, 1074, 3]) # TODO: Kill me
    # print(padded_image.get_shape(), image.get_shape(), mask.get_shape())

    disc_real_net, disc_real_logits = MolanetPoc._create_discriminator(image_patches, mask_patches)
    gen_net, _ = MolanetPoc._create_generator(image)
    gen_net_use, _ = MolanetPoc._create_generator(image, reuse=True, reshape=True)
    # gen_net = tf.reshape(gen_net, [1, 767, 1022, 1]) # TODO: Kill me

    disc_fake_net, disc_fake_logits = MolanetPoc._create_discriminator(image_patches, gen_net, reuse=True)

    loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits)))
    loss_disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_logits, labels=tf.ones_like(disc_real_logits)))
    loss_disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
    loss_disc = loss_disc_real + loss_disc_fake

    # Optimizers
    optim_disc = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
    optim_gen = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)

    trainable_variables = tf.trainable_variables()
    vars_disc = [var for var in trainable_variables if var.name.startswith("discriminator")]
    vars_gen = [var for var in trainable_variables if var.name.startswith("generator")]

    update_disc = optim_disc.minimize(loss_disc, var_list=vars_disc)
    update_gen = optim_gen.minimize(loss_gen, var_list=vars_gen)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        create_gen_patches = MolanetPoc._pad_image(tf.convert_to_tensor(in_image), tf.TensorShape([1, 94, 94, 3]), tf.TensorShape([1, 70, 70, 1]))
        create_gen_patches = tf.extract_image_patches(create_gen_patches, [1, 94, 94, 1], [1, 70, 70, 1], rates=[1, 1, 1, 1], padding="VALID")
        create_gen_patches = tf.reshape(create_gen_patches, [-1, 94, 94, 3])

        patches_image = sess.run(MolanetPoc.split_discriminator_image(in_image))
        patches_segmentation = sess.run(MolanetPoc.split_discriminator_image(in_segmentation))
        patches_gen_image = sess.run(create_gen_patches)

        Image.fromarray(np.reshape((patches_image[20] + 1.0) / 2.0 * 255.0, (patches_image.shape[1], patches_image.shape[2], 3)).astype(np.uint8)).save("/tmp/patches_image.png")
        Image.fromarray(np.reshape((patches_segmentation[20] + 1.0) / 2.0 * 255.0, (patches_segmentation.shape[1], patches_segmentation.shape[2])).astype(np.uint8)).save("/tmp/patches_segmentation.png")
        Image.fromarray(np.reshape((patches_gen_image[20] + 1.0) / 2.0 * 255.0, (patches_gen_image.shape[1], patches_gen_image.shape[2], 3)).astype(np.uint8)).save("/tmp/patches_gen_image.png")

        print("Starting training...")
        for epoch in range(3000):
            _, cost_disc = sess.run([update_disc, loss_disc], feed_dict={image_patches: patches_image, image: patches_gen_image, mask_patches: patches_segmentation})
            _, cost_gen = sess.run([update_gen, loss_gen], feed_dict={image: patches_gen_image, image_patches: patches_image})

            if epoch % 1 == 0:
                print(f"{epoch} losses: gen={cost_gen:.8f}, disc={cost_disc:.8f}")

            # if epoch % 1 == 0:
            #     sample = sess.run(gen_net_use, feed_dict={image: patches_gen_image})
            #
            #     Image.fromarray(np.reshape((sample + 1.0) / 2.0 * 255.0, (sample.shape[1], sample.shape[2])).astype(np.uint8)).save(f"/tmp/res_{epoch}.png")
            #     print("Saved image")

            if epoch % 10 == 0:
                sample = sess.run(gen_net, feed_dict={image: patches_gen_image})
                result = np.zeros([11 * 70, 15 * 70])
                for y in range(result.shape[0] // 70):
                    for x in range(result.shape[1] // 70):
                        y_offset = y * 70
                        x_offset = x * 70

                        result[y_offset:y_offset+70, x_offset:x_offset+70] = sample[y * 15 + x, :, :, 0]

                Image.fromarray(((result + 1.0) / 2.0 * 255.0).astype(np.uint8)).save(f"/tmp/res_full_{epoch}.png")
                print("Saved big image")
