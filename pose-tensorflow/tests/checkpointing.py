import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp


def toy_varsave(save_path):
    # Create some variables.
    v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)
    
    inc_v1 = v1.assign(v1+1)
    dec_v2 = v2.assign(v2-1)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    with tf.Session() as sess:
        sess.run(init_op)
        # Do some work with the model.
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())
        print("computation")
        inc_v1.op.run()
        dec_v2.op.run()
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())
        # Save the variables to disk.
        saver.save(sess, save_path)
        print("Model saved in path: %s" % save_path)
        

def toy_varload(save_path):
    tf.reset_default_graph()
    
    # Create some variables.
    v1 = tf.get_variable("v1", shape=[3])
    v2 = tf.get_variable("v2", shape=[5])
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, save_path)
        print("Model restored.")
        # Check the values of the variables
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())


def toy_run(save_path):
    print("starting toy_varsave")
    toy_varsave(save_path)
    print("finished toy_varsave")
                      
    print("starting toy_varload")
    toy_varload(save_path)
    print("finished toy_varload")


def examine_ckpt(ckpt_filename)
# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]

# print only tensor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load DeepLabCut network')

    parser.add_argument('--ckpt_fileanme',
                        default='/tmp/model.ckpt',
                        help='Save path')
    
    args = parser.parse_args()
    
    toy_run(args.ckpt_filename)
