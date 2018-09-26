import time
import logging
import threading

import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import load_config
from dataset.factory import create as create_dataset
from nnet.net_factory import pose_net
from nnet.pose_net import get_batch_spec
from util.logging import setup_logging

from tests.checkpointing import toy_run

class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg.multi_step
        self.current_step = 0

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1

        return lr


def setup_preloading(batch_spec):
    placeholders = {name: tf.placeholder(tf.float32, shape=spec) for (name, spec) in batch_spec.items()}
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20

    q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()

    batch = {}
    for idx, name in enumerate(names):
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    return batch, enqueue_op, placeholders


def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    while not coord.should_stop():
        batch_np = dataset.next_batch()
        food = {pl: batch_np[name] for (name, pl) in placeholders.items()}
        sess.run(enqueue_op, feed_dict=food)


def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = tf.train.Coordinator()

    t = threading.Thread(target=load_and_enqueue,
                         args=(sess, enqueue_op, coord, dataset, placeholders))
    t.start()

    return coord, t


def get_optimizer(loss_op, cfg):
    learning_rate = tf.placeholder(tf.float32, shape=[])

    global_step = tf.Variable(0, name='global_step', trainable=False)
    # train_op = optimizer.minimize(loss, global_step=global_step)
    
    if cfg.optimizer == "sgd":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif cfg.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(cfg.adam_lr)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer, global_step=global_step)
    # train_op = slim.learning.create_train_op(loss_op, optimizer)
    
    return learning_rate, train_op


def train():
    # for cluster
    start = time.time()
    
    setup_logging()

    cfg = load_config()
    dataset = create_dataset(cfg)

    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    losses = pose_net(cfg).train(batch)
    total_loss = losses['total_loss']

    for k, t in losses.items():
        tf.summary.scalar(k, t)
    merged_summaries = tf.summary.merge_all()

    variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(max_to_keep=5)

    sess = tf.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)

    train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

    learning_rate, train_op = get_optimizer(total_loss, cfg)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    ckpt_filename = tf.train.latest_checkpoint('./')
    if ckpt_filename is None:
        print("restoring initial weights")
        restorer.restore(sess, cfg.init_weights)
    else:
        print("restore from checkpoint file ", ckpt_filename)
        restorer.restore(sess, ckpt_filename)

    max_iter = int(cfg.multi_step[-1][1])

    display_iters = cfg.display_iters
    cum_loss = 0.0
    lr_gen = LearningRate(cfg)

    step_tensor = tf.train.get_global_step()
    step = tf.train.global_step(sess, step_tensor)

    print("current step: ", step)
    limits = [ lr_gen.steps[i][1] for i in range(len(lr_gen.steps)) ]
    idx = sum( [limits[i] < step for i in range(len(limits))] )
    lr_gen.current_step = max(0, idx-1)

    for it in range(step+1, max_iter+1):
        current_lr = lr_gen.get_lr(it)
        [_, loss_val, summary, step] = sess.run([train_op, total_loss, merged_summaries, step_tensor],
                                          feed_dict={learning_rate: current_lr})
        # print("step, it: ", step, it)
        cum_loss += loss_val
        train_writer.add_summary(summary, it)

        if it % display_iters == 0:
            average_loss = cum_loss / display_iters
            cum_loss = 0.0
            logging.info("iteration: {} loss: {} lr: {}"
                         .format(it, "{0:.4f}".format(average_loss), current_lr))

        # Save snapshot
        if (it % cfg.save_iters == 0 and it != 0) or it == max_iter:
            model_name = cfg.snapshot_prefix
            saver.save(sess, model_name, global_step=step_tensor)

        cur = time.time()
        if cur - start > 60: # if more than 55 minutes
            print("exiting!")
            model_name = cfg.snapshot_prefix
            print(tf.train.global_step(sess, step_tensor))
            saver.save(sess, model_name, global_step=step_tensor)
            break

    sess.close()
    coord.request_stop()
    coord.join([thread])


def test(config_filename, save_path):
    # for cluster
    start = time.time()
    
    setup_logging()

    cfg = load_config()
    dataset = create_dataset(cfg)

    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    losses = pose_net(cfg).train(batch)
    total_loss = losses['total_loss']

    for k, t in losses.items():
        tf.summary.scalar(k, t)
    merged_summaries = tf.summary.merge_all()

    variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])

    learning_rate, train_op = get_optimizer(total_loss, cfg)

    gs = slim.get_variables_by_name("global_step")[0]

    saver = tf.train.Saver(max_to_keep=5)

    ckpt_filename = tf.train.latest_checkpoint(save_path)
    if ckpt_filename is None:
        print("restoring initial weights")
        restorer = tf.train.Saver(variables_to_restore)
        restore_filename = cfg.init_weights
    else:
        print("restore from checkpoint file ", ckpt_filename)
        restorer = tf.train.Saver(variables_to_restore.append(gs))
        restore_filename = ckpt_filename

    max_iter = int(cfg.multi_step[-1][1])

    display_iters = cfg.display_iters
    cum_loss = 0.0
    lr_gen = LearningRate(cfg)

    with tf.Session() as sess:
        print("in session...")

        sess.run(tf.global_variables_initializer())
        print("variables initialized...")

        coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)
        train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

        restorer.restore(sess, restore_filename)
        print("variables restored...")

        step = tf.train.global_step(sess, gs)
        print("current step: ", step)
        
        limits = [ lr_gen.steps[i][1] for i in range(len(lr_gen.steps)) ]
        idx = sum( [limits[i] < step for i in range(len(limits))] )
        lr_gen.current_step = max(0, idx-1)
        print("current lr: ", lr_gen.get_lr(step))
        
        for it in range(step, max_iter+1):
            current_lr = lr_gen.get_lr(it)
            [_, loss_val, summary, gs_] = sess.run([train_op, total_loss, merged_summaries, gs],
                                                    feed_dict={learning_rate: current_lr})

            cum_loss += loss_val
            train_writer.add_summary(summary, it)

            if it % display_iters == 0:
                average_loss = cum_loss / display_iters
                cum_loss = 0.0
                logging.info("iteration: {} loss: {} lr: {}"
                             .format(it, "{0:.4f}".format(average_loss), current_lr))

            # Save snapshot
            if (it % cfg.save_iters == 0 and it != 0) or it == max_iter:
                model_name = cfg.snapshot_prefix
                saver.save(sess, model_name, global_step=gs)

            cur = time.time()
            if cur - start > 50: # if more than 55 minutes
                print("exiting!")
                model_name = cfg.snapshot_prefix
                print("current global step: ", tf.train.global_step(sess, step_tensor))
                saver.save(sess, model_name, global_step=gs)
                return



            # if it % 10 == 0:
            #     print("saving...")
            #     saver.save(sess, save_path + 'testing', global_step=gs)
            #     return
                



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load DeepLabCut network')

    parser.add_argument('--config_filename',
                        default='models/reachingJan30-trainset95shuffle1/train/pose_cfg.yaml',
                        help='Configuration file location')

    parser.add_argument('--save_path',
                        default='./',
                        help='Save path')
    
    # train()
    args = parser.parse_args()
    
    test(args.config_filename, args.save_path)

