import os.path as osp

import tensorflow as tf
import numpy as np
import h5py
import scipy.misc
import skimage as skim
import skimage.io as skio
from tqdm import tqdm

from px2graph.util import setup, img
from px2graph.opts import parse_command_line
from px2graph.util import viz


def main():

    # Initial setup
    opt = parse_command_line()
    train_flag = tf.placeholder(tf.bool, [])
    task, loader, inp, label, sample_idx, ds = setup.init_task(opt, train_flag, return_dataset=True)
    net, loss, pred, accuracy, optim, lr = setup.init_model(opt, task, inp, label,
                                                            sample_idx, train_flag)

    # Prepare TF session
    # summaries, image_summaries = task.setup_summaries(net, inp, label, loss, pred, accuracy)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())

    # Start data loading threads
    loader.start_threads(sess)

    # Restore previous session if continuing experiment
    # Load pretrained weights
    if opt.restore_session is not None:
        print("Restoring previous session...",'(exp/' + opt.restore_session + '/snapshot)')
        if opt.new_optim:
            # Optimizer changed, don't load values associated with old optimizer
            tmp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            tmp_saver = tf.train.Saver(tmp_vars)
            tmp_saver.restore(sess, 'exp/' + opt.restore_session + '/snapshot')
        else:
            saver.restore(sess, 'exp/' + opt.restore_session + '/snapshot')

    # Generate predictions
    num_samples = opt.iters['valid'] * opt.batchsize
    split = opt.predict
    idxs = opt.idx_ref[split]
    num_samples = idxs.shape[0]

    pred_dims = {k:[int(d) for d in pred[k].shape[1:]] for k in pred}
    final_preds = {k:np.zeros((num_samples, *pred_dims[k])) for k in pred}
    idx_ref = np.zeros(num_samples)
    flag_val = False

    print("Generating predictions...")
    loader.start_epoch(sess, split, train_flag, num_samples, flag_val=flag_val, in_order=True)
    for _ in range(5):
        tmp_idx, tmp_pred, tmp_inp, tmp_label = sess.run(
            [sample_idx, pred, inp, label], feed_dict={train_flag: flag_val})
        assert len(tmp_idx) == 1, "Only support batchsize 1 for visualizing the predictions"

        tmp_idx = tmp_idx[0][0]
        img_id = ds.get_id(tmp_idx)
        print('sample index: %d, image_name: %d.jpg' % (tmp_idx, img_id))

        inp_img = skim.img_as_ubyte(tmp_inp[0][0])
        viz.visualize_preds(inp_img, tmp_pred['objs'][0], tmp_pred['rels'][0], ds,
                            str(img_id), osp.join('exp', opt.exp_id))


if __name__ == '__main__':
    main()
