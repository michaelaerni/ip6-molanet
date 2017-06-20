import tensorflow as tf


class NetworkFactory(object):
    """
    Factory for generator and discriminator networks.

    This is an abstract base class.
    Child classes should inherit override the network creation methods.
    Additional network configuration should be supplied in the child's constructor.
    """

    def create_generator(
            self,
            source_tensor: tf.Tensor,
            reuse: bool = False,
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates a generator network and optionally applies summary options where useful.

        :param source_tensor: Input for the created generator
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param apply_summary: If True, summary operations will be added to the network. Defaults to True.
        :return: Output tensor of the created generator
        """

        raise NotImplementedError("This method should be overridden by child classes")

    def create_discriminator(
            self,
            source_tensor: tf.Tensor,
            target_tensor: tf.Tensor,
            reuse: bool = False,
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates a discriminator network and optionally applies summary options where useful.

        :param source_tensor: Input tensor for the corresponding generator
        :param target_tensor: Tensor of the generated value for the input source_tensor
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param apply_summary: If True, summary operations will be added to the network. Defaults to True.
        :return: Output tensor of the created discriminator as unscaled logits
        """

        raise NotImplementedError("This method should be overridden by child classes")
