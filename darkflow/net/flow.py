import os
import time
import numpy as np
import pickle

import cv2
import tensorflow as tf
from multiprocessing.pool import ThreadPool

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self, test_func=None):
    from time import localtime, strftime

    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    input_width = self.meta['net']['width']
    input_height = self.meta['net']['height']
    prob_img = np.zeros(shape=(input_width, input_height))

    fix_train_count = 0
    if 'train_layer_limit' in self.meta:
        fix_train_count = self.meta['train_layer_limit'] - self.FLAGS.load

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key]
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        # Make body, face probability map for customized training
        if hasattr(self, 'partInfoList') and len(self.partInfoList) > 0:
            confs = datum['confs']
            coord = datum['coord']
            probs = datum['probs']

            for info in self.partInfoList:
                grid_x = info.place_holder.shape[1].value
                grid_y = info.place_holder.shape[2].value
                cell_width = input_width // grid_x
                cell_height = input_height // grid_y
                prob_batch = np.zeros(shape=(x_batch.shape[0], grid_x, grid_y, 1))

                # print('Class No %d' % info.class_no)
                for ei in range(x_batch.shape[0]):
                    img = x_batch[ei].copy()
                    prob_img.fill(0)

                    # For display (grid)
                    for gi in range(grid_x):
                        for gj in range(grid_y):
                            x = int(gi * cell_width)
                            y = int(gj * cell_width)
                            cv2.rectangle(img, (x, y), (x+cell_width, y+cell_width), (0,0,0), 1)

                    for xi in range(grid_x * grid_y):
                        cf = confs[ei][xi][0]
                        cls = np.argmax(probs[ei][xi])

                        if cf > 0:
                            cell_idx_x = int(xi % grid_x)
                            cell_idx_y = int(xi // grid_y)

                            # Visualization code for debugging
                            cd = coord[ei][xi][0]

                            cell_pos_x = (cell_idx_x * cell_width) + (cd[0] * cell_width)
                            cell_pos_y = (cell_idx_y * cell_height) + (cd[1] * cell_height)

                            w = int(cd[2] * cd[2] * input_width)
                            h = int(cd[3] * cd[3] * input_height)
                            x = int(cell_pos_x - (w/2))
                            y = int(cell_pos_y - (h/2))

                            if cls == info.class_no:
                                cv2.rectangle(prob_img, (x, y), (x + w, y + h), (255, 255, 255), -1)

                    # cv2.imshow('Input', img)
                    # cv2.imshow('Map', prob_img)
                    # cv2.waitKey()

                    cv2.normalize(prob_img, prob_img, 0, 1, cv2.NORM_MINMAX)
                    cv2.resize(prob_img, dsize=(grid_x, grid_y), dst=prob_batch[ei])
                    feed_dict[info.place_holder] = prob_batch

                    # cv2.resize(part_prob[ei], dsize=(img_width, img_width), dst=prob_img)
                    # cv2.imshow('Resized', prob_img)
                    # cv2.waitKey()

        part_loss_fetched = None

        if fix_train_count > 0:
            train_op = self.train_op2
            fix_train_count = fix_train_count - 1
            if fix_train_count <= 0:
                print('Fixed train mode is finished')
        else:
            train_op = self.train_op

        if hasattr(self, 'partInfoList') and len(self.partInfoList) > 0:
            # Minimize Part Probability Error
            part_train_ops = []
            for info in self.partInfoList:
                part_train_ops.append(info.train_op)
            self.sess.run(part_train_ops, feed_dict)

            # Calculate Part Probability Loss
            part_loss_ops = []
            for info in self.partInfoList:
                part_loss_ops.append(info.loss)
            part_loss_fetched = self.sess.run(part_loss_ops, feed_dict)

            # Minimize Regression Error
            fetches = [train_op, loss_op, self.summary_op]
            fetched = self.sess.run(fetches, feed_dict)
        else:
            fetches = [train_op, loss_op, self.summary_op]
            fetched = self.sess.run(fetches, feed_dict)

        loss = fetched[1]
        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        # self.writer.add_summary(fetched [2], step_now)

        if part_loss_fetched is not None:
            part_loss_class = [0.0, 0.0]
            for idx, info in enumerate(self.partInfoList):
                if info.class_no == 0:
                    part_loss_class[0] += part_loss_fetched[idx]
                elif info.class_no == 1:
                    part_loss_class[1] += part_loss_fetched[idx]
            form = '%s   step %d - loss %.3e (%.3e/%.3e) - moving ave loss %.3e' % \
                   (strftime("%Y-%m-%d %H:%M:%S", localtime()), step_now, loss,
                    part_loss_class[0], part_loss_class[1], loss_mva)
            self.say(form)
        else:
            form = '{}   step {} - loss {} - moving ave loss {}'
            self.say(form.format(strftime("%Y-%m-%d %H:%M:%S", localtime()), step_now,
                                 loss, loss_mva))


        profile += [(loss, loss_mva)]

        #ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        ckpt = not ((step_now % self.FLAGS.save) == 0)

        args = [step_now, profile]
        if not ckpt:
            _save_ckpt(self, *args)

        if (self.FLAGS.test_step > 0) and (step_now % self.FLAGS.test_step == 0):
            test_func(step_now)

    if ckpt:
        _save_ckpt(self, *args)


def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    if True:
        ''' 
        if not hasattr(self, 'partial_run_handle'):
            self.partial_run_handle = self.sess.partial_run_setup(self.tf_layers, [self.inp])
        temp = self.sess.partial_run(self.partial_run_handle, self.tf_layers[2], feed_dict=feed_dict)
        out = self.sess.partial_run(self.partial_run_handle, self.out)[0]
        
        temp, out = self.sess.partial_run(self.partial_run_handle, [self.tf_layers[2], self.out], feed_dict=feed_dict)[0]
        out = out[0]                
        '''
        self.intermediate_feature, out = self.sess.run([self.tf_layers[2], self.out], feed_dict=feed_dict)
        out = out[0]

    else:
        out = self.sess.run(self.out, feed_dict)[0]

    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

def return_features(self, boxesInfo, grid):
    inter_f = self.intermediate_feature
    f_shape = inter_f.shape
    resize_ratio = f_shape[1] / 608
    features = []

    for r in boxesInfo:
        x1 = r[0]
        y1 = r[1]
        x2 = r[0] + r[2]
        y2 = r[1] + r[3]

        x1 = math.ceil(x1 * resize_ratio)
        y1 = math.ceil(y1 * resize_ratio)
        x2 = math.ceil(x2 * resize_ratio)
        y2 = math.ceil(y2 * resize_ratio)

        w = x2 - x1
        h = y2 - y1

        grid_w = int(w / grid[0])
        grid_h = int(h / grid[1])

        if grid_w <= 0:
            grid_w = 1
        if grid_h <= 0:
            grid_h = 1

        if grid[2] <= 0 :
            grid[2] = f_shape[3]

        grid_c = int(f_shape[3] / grid[2])
        if grid_h <= 0:
            grid_c = 1

        f = np.zeros(shape=grid, dtype=np.float32)
        for i in range(grid[0]):
            for j in range(grid[1]):
                for k in range(grid[2]):
                    s1 = i * grid_w
                    e1 = s1 + grid_w

                    if e1 >= f_shape[1]:
                        continue

                    s2 =  j * grid_h
                    e2 = s2 + grid_h

                    if e2 >= f_shape[2]:
                        continue

                    s3 = k * grid_c
                    e3 = s3 + grid_c

                    if e3 >= f_shape[3]:
                        continue

                    val = np.mean(inter_f[0, s1:e1, s2:e2, s3:e3])
                    f[i,j,k] = val
                    if math.isnan(val):
                        raise Exception('Something wrong happend! value is nan!')

        f = np.reshape(f, newshape=(grid[0] * grid[1] * grid[2]))
        features.append(f)
    return features

import math

def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        inp_feed = list(); new_all = list()
        this_batch = all_inps[from_idx:to_idx]
        for inp in this_batch:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        this_batch = new_all

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
