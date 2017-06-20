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
            x: tf.Tensor,
            reuse: bool = False,
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates a generator network and optionally applies summary options where useful.

        :param x: Input for the created generator
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :return: Output tensor of the created generator
        """

        raise NotImplementedError("This method should be overridden by child classes")

    def create_discriminator(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            reuse: bool = False,
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates a discriminator network and optionally applies summary options where useful.

        :param x: Input tensor for the corresponding generator
        :param y: Tensor of the generated or real value for the input x
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :return: Output tensor of the created discriminator as unscaled logits
        """

        raise NotImplementedError("This method should be overridden by child classes")


class ObjectiveFactory(object):
    """
    Factory for generator and discriminator objectives (cost functions).

    This is an abstract base class.
    Child classes should inherit override the objective creation methods.
    Additional network configuration should be supplied in the child's constructor.
    """

    def create_generator_loss(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            generator: tf.Tensor,
            generator_discriminator: tf.Tensor,
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates the generator loss function and optionally applies summary options.

        :param x: Input tensor which is fed to the generator
        :param y: Ground truth output for the given x
        :param generator: Generated output for the given x
        :param generator_discriminator: Discriminator logits for the generated output corresponding to the x
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :return: Loss function for the generator which can be used for optimization
        """

        raise NotImplementedError("This method should be overridden by child classes")

    def create_discriminator_loss(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            generator_discriminator: tf.Tensor,
            real_discriminator: tf.Tensor,
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates the discriminator loss function and optionally applies summary options.

        :param x: Input tensor which is fed to the generator
        :param y: Ground truth output for the given x
        :param generator_discriminator: Discriminator logits for the generated output corresponding to the x
        :param real_discriminator: Discriminator logits for the ground truth output
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :return: Loss function for the discriminator which can be used for optimization
        """

        raise NotImplementedError("This method should be overridden by child classes")
