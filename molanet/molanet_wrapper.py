import argparse
import os
import sys

import tensorflow as tf


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Molanet Wrapper script")

    parser.add_argument("argsfile", type=str,
                        help="path to the file containing arguments for main script")
    parser.add_argument("--shutdown", type=bool, default=True,
                        help="Send shutdown command after completion or on Error")
    return parser


def check_tensorflow_gpu():
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    with tf.Session() as sess:
        sess.run(c)

    print("tensorflow gpu seems to be working")


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    shutdown_on_exit = args.shutdown
    if shutdown_on_exit and not os.name == 'nt' and os.geteuid() != 0:
        exit("You need to have root privileges to shutdown on exit.\n"
             "Please try again, this time using 'sudo'. Exiting.")
    elif not shutdown_on_exit:
        print("This machine will not automatically shutdown. DO NOT FORGET!")

    # make molanet available if it isn't already. this is a bit hacky
    path_to_this_script = sys.argv[0]
    path_to_module = path_to_this_script.replace("/molanet/molanet_wrapper.py", "")
    if not path_to_module in sys.path:
        sys.path.append(path_to_module)

    from molanet.models.molanet_poc import molanet_main

    check_tensorflow_gpu()

    # read string arguments to molanet
    with open(args.argsfile) as f:
        molanet_args = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        molanet_args = [x.strip() for x in molanet_args]


    def ask_shutdown():
        if shutdown_on_exit and not os.name == "nt":
            print("exiting with shutdown")
            os.system("shutdown now -h")


    try:
        print("running molanet")
        molanet_main(molanet_args)

        ask_shutdown()
    except TypeError as e:
        print(e)
        print("TypeError - probably invalid args")
    except Exception as e:
        print(e)
        ask_shutdown()
