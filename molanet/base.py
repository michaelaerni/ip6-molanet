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
            input_tensor: tf.Tensor,
            reuse: bool = False,
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates a generator network and optionally applies summary options where useful.

        :param input_tensor: Input of the created generator
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param apply_summary: If True, summary operations will be added to the network. Defaults to True.
        :return: Output tensor of the created generator
        """

        raise NotImplementedError("This method should be overridden by child classes")

    def create_discriminator(
            self,
            input_tensor: tf.Tensor,
            reuse: bool = False,
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates a discriminator network and optionally applies summary options where useful.

        :param input_tensor: Input of the created discriminator
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param apply_summary: If True, summary operations will be added to the network. Defaults to True.
        :return: Output tensor of the created discriminator
        """

        raise NotImplementedError("This method should be overridden by child classes")
