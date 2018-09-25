import os

import tensorflow as tf
import numpy as np

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


def examine_ckpt(ckpt_path, var_name=None):
    end_matter = len(ckpt_path.split('/')[-1])
    ckpt_dir = ckpt_path[:-end_matter]
    print(end_matter, ckpt_dir)

    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_file is None:
        ckpt_file = ckpt_path if os.path.exists(ckpt_path) else ckpt_file
        
    # print all tensors in checkpoint file
    if var_name is None:
        chkp.print_tensors_in_checkpoint_file(ckpt_file, tensor_name='', all_tensors=True)
    else:
        chkp.print_tensors_in_checkpoint_file(ckpt_file, tensor_name=var_name, all_tensors=False)
    

def generate_linear_data(a, b, ub, lb, num=50, eps=1e-2):
    '''
    generate a dataset for line fitting.

    Input:
    a - scalar value slope
    b - scalar value offset
    ub - scalar value upper limit for independent variable x
    lb - scalar value lower limit for independent variable x
    eps - scalar value noise

    Output:
    x - (num, ) independent variable
    y - (num, ) dependent variable
    '''

    x = np.linspace(lb, ub, num=num)
    y = a*x + b + np.random.rand(num)*eps

    return x, y


def fit_data(x, y, lr, max_iter, print_iter, ckpt_iter, ckpt_path):
    '''
    fits (x,y) data with a linear model using gradient descent

    Inputs:
    x - (num, ) observation
    y - (num, ) labels
    '''
    # variables for slope and offset
    a = tf.get_variable("a", shape=[1], initializer=tf.ones_initializer())
    b = tf.get_variable("b", shape=[1], initializer=tf.ones_initializer())
    
    y_hat = tf.add( tf.multiply(a, x), b)
    loss = tf.reduce_sum(tf.square(y - y_hat))

    global_step = tf.Variable(0, trainable=False, name='global_step')
    train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for it in range(max_iter):
            _, loss_, gs, a_, b_ = sess.run([train_op, loss, global_step, a, b])

            if it % print_iter == 0:
                print("global_step: {}, loss: {}, slope: {}, offset: {}".format(gs, loss_, a_, b_))

            if it % ckpt_iter == 0:
                saver.save(sess, ckpt_path, global_step=global_step)




if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load DeepLabCut network')
    # checkpoint save path
    parser.add_argument('--ckpt_filename',
                        default='/tmp/model.ckpt',
                        help='Save path')
    # learning rate
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate for optimizer')

    # max iterations
    parser.add_argument('--max_iter',
                        type=int,
                        default=1000,
                        help='Maximum iterations for optimizer')

    # print iteration
    parser.add_argument('--print_iter',
                        type=int,
                        default=100,
                        help='Print loss every print_iter iterations')

    # checkpoint iteration
    parser.add_argument('--ckpt_iter',
                        type=int,
                        default=500,
                        help='Save variables every ckpt_iter iterations')

    # data generation
    # slope
    parser.add_argument('--slope',
                        type=float,
                        default=5,
                        help='Slope parameter for generated data')

    # offset
    parser.add_argument('--offset',
                        type=float,
                        default=2,
                        help='Offset parameter for generated data')

    # upper bound
    parser.add_argument('--upper_bound',
                        type=float,
                         default=10,
                         help='Upper bound for independent parameter')

    # lower bound
    parser.add_argument('--lower_bound',
                        type=float,
                        default=-10,
                        help='Lower bound for independent paramenter')

    # noise
    parser.add_argument('--noise',
                        type=float,
                        default=1,
                        help='Noise parameter for data generation')
    
    args = parser.parse_args()
    
    # toy_run(args.ckpt_filename)

    # x, y = generate_linear_data(a=args.slope,
    #                             b=args.offset,
    #                             ub=args.upper_bound,
    #                             lb=args.lower_bound,
    #                             eps=args.noise)

    # fit_data(x,
    #          y,
    #          lr=args.learning_rate,
    #          max_iter=args.max_iter,
    #          print_iter=args.print_iter,
    #          ckpt_iter=args.ckpt_iter,
    #          ckpt_path=args.ckpt_filename)

    # examine_ckpt(args.ckpt_filename)
    examine_ckpt(args.ckpt_filename, var_name='global_step')
