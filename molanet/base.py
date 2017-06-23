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
            generator: tf.Tensor,
            generator_discriminator: tf.Tensor,
            real_discriminator: tf.Tensor,
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates the discriminator loss function and optionally applies summary options.

        :param x: Input tensor which is fed to the generator
        :param y: Ground truth output for the given x
        :param generator: Generated output for the given x
        :param generator_discriminator: Discriminator logits for the generated output corresponding to the x
        :param real_discriminator: Discriminator logits for the ground truth output
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :return: Loss function for the discriminator which can be used for optimization
        """

        raise NotImplementedError("This method should be overridden by child classes")


class NetworkTrainer(object):
    # TODO: Doc

    def __init__(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            network_factory: NetworkFactory,
            objective_factory: ObjectiveFactory,
            learning_rate: float):
        # Create networks
        self._generator = network_factory.create_generator(x)
        self._discriminator_generated = network_factory.create_discriminator(x, self._generator)
        self._discriminator_real = network_factory.create_discriminator(x, y, reuse=True)

        # Create losses
        self._generator_loss = objective_factory.create_generator_loss(
            x, y, self._generator, self._discriminator_generated)
        self._discriminator_loss = objective_factory.create_discriminator_loss(
            x, y, self._generator, self._discriminator_generated, self._discriminator_real)

        # Create optimizers
        trainable_variables = tf.trainable_variables()
        variables_discriminator = [var for var in trainable_variables if var.name.startswith("discriminator")]
        variables_generator = [var for var in trainable_variables if var.name.startswith("generator")]

        self._optimizer_generator = tf.train.AdamOptimizer(learning_rate, beta1=0.5, name="adam_generator")
        self._optimizer_discriminator = tf.train.AdamOptimizer(learning_rate, beta1=0.5, name="adam_discriminator")

        self._op_generator = self._optimizer_generator.minimize(self._generator_loss, var_list=variables_generator)
        self._op_discriminator = self._optimizer_discriminator.minimize(self._discriminator_loss, var_list=variables_discriminator)

        # Iteration counter
        self._global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.uint16)

    def train(self, sess: tf.Session, summary_directory: str):
        # TODO: Remove prints everywhere

        # TODO: Better summary handling
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(summary_directory, sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        step_update = tf.assign_add(self._global_step, 1)

        try:
            while not coord.should_stop():
                _, _, current_summary, _, iteration = sess.run([
                    self._op_discriminator,
                    self._op_generator,
                    summary,
                    step_update,
                    self._global_step])
                summary_writer.add_summary(current_summary, iteration)
                print(f"Iteration {iteration} done")

        except tf.errors.OutOfRangeError:
            print("Epoch limit reached, training stopped")
        finally:
            coord.request_stop()

        print("Waiting for threads to finish...")
        coord.join(threads)

        print("Training finished")
