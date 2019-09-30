import random
import math

import numpy as np
from scipy import optimize
import pickle

import tensorflow as tf
import tensorlayer as tl

import elm.DataInput as DataInput
import elm.Utilities as util

class ELM:
    OPTION_TF_SOLVER_LEVEL = 1

    def __init__(self, w1=None, w2=None):
        self.__w1 = w1
        self.__w2 = w2
        self.__b1 = self.__b2 = None

        self.use_L1_Panelty = False
        self.use_double = False
        self.activation = 'Sigmoid'
        self.activation_normalizer = None
        self.output_activation = ''
        self.sample_weight = None

        self.store_h = False
        self.__h = self.__p = None
        self.batch_size = 100
        self.display_step = 0
        self.train_param = None
        self.device = '/gpu:0'
        self.device_log = False

        self.sess = self.inp = self.out = self.phase_train = self.label = self.correct_sum = None

        # Experimental - FAILED
        self.use_af_solver = False
        self.use_tf_solver = False
        self.use_tf_solver_reuse = False
        self.tf_solver_ctx = None

        if self.__w1 is not None:
            self.__b1 = np.full([self.__w1.shape[1]], 0, dtype=self.__w1.dtype)
        if self.__w2 is not None:
            self.__b2 = np.full([self.__w2.shape[0]], 0, dtype=self.__w2.dtype)

    @property
    def transformed(self):
        return self.__h

    @property
    def input_weight(self):
        return self.__w1

    @property
    def input_bias(self):
        return self.__b1

    @property
    def output_weight(self):
        return self.__w2

    @property
    def output_bias(self):
        return self.__b2

    @property
    def train_info(self):
        class train_param:
            pass

        p = train_param()
        p.activation = self.activation
        p.use_double = self.use_double
        return p

    def init_weight(self, data_set, n_hidden, init_method='Normal', init_param=None, train_param=None,
                    w_normalization=None, n_fdim=-1, display_step=0):

        if data_set is not None:
            fdim = data_set.train.dim_features

        if (n_fdim != -1):
            fdim = n_fdim
        w1, b1 = ELM.create_init_weight(data_set, fdim, n_hidden, init_method, init_param, train_param,
                                        w_normalization, display_step=display_step)

        self.__w1 = w1
        self.__b1 = b1

    def set_input_weight(self, w1, b1=None):
        self.__w1 = w1
        self.__b1 = b1

        if b1 is None:
            b1 = np.full([w1.shape[1]], 0)
            self.__b1 = b1.astype(np.float32)
            pass

    def build_model(self, x, w1=None, b1=None, w2=None, b2=None):

        # By default, use stored parameters for simplest method call
        if w1 is None:
            w1 = self.__w1
        if b1 is None:
            b1 = self.__b1
        if w2 is None:
            w2 = self.__w2
        if b2 is None:
            b2 = self.__b2

        layer1 = tf.matmul(x, w1) + b1

        if self.activation == 'Sigmoid':
            layer1 = tf.nn.sigmoid(layer1)
        elif self.activation == 'ReLU':
            layer1 = tf.nn.relu(layer1)
        elif self.activation == 'ExpLU':
            layer1 = tf.nn.elu(layer1)
        elif self.activation == 'LeakyReLU':
            layer1 = tf.nn.leaky_relu(layer1, 0.1)
        elif self.activation == 'Linear':
            pass
        else:
            raise Exception('Invalid activation = %s' + self.activation)

        if self.activation_normalizer is not None:
            layer1 = self.activation_normalizer.normalize(layer1)

        if self.phase_train is not None:
            layer1 = tf.contrib.layers.batch_norm(layer1, center=True, scale=True, is_training=self.phase_train, scope='bn1')
            # layer1 = batch_norm(layer1, layer1.shape[1], self.phase_train)

        layer2 = tf.matmul(layer1, w2) + b2
        if self.output_activation == 'Sigmoid':
            layer2 = tf.nn.sigmoid(layer2)
        elif self.output_activation == 'ReLU':
            layer2 = tf.nn.relu(layer2)
        elif self.activation == 'ExpLU':
            layer1 = tf.nn.elu(layer2)
        elif self.output_activation == 'LeakyReLU':
            layer2 = tf.nn.leaky_relu(layer2, 0.1)

        return layer2

    def do_transform(self, x, data=None, before_activation=False):
        if data is None:
            h1 = tf.matmul(x, self.__w1) + self.__b1

            if not before_activation:
                if self.activation == 'Sigmoid':
                    h1 = tf.nn.sigmoid(h1)
                elif self.activation == 'ReLU':
                    h1 = tf.nn.relu(h1)
                elif self.activation == 'ExpLU':
                    layer1 = tf.nn.elu(h1)
                elif self.activation == 'LeakyReLU':
                    h1 = tl.activation.leaky_relu(h1, 0.1)
                elif self.activation == 'Linear':
                    pass
                else:
                    raise Exception('Invalid activation = %s' % self.activation)
            return h1
        else:
            n_input = data.dim_features
            n_classes = data.dim_labels
            batch_size = self.batch_size

            if batch_size > data.num_examples:
                batch_size = data.num_examples

            d_type = tf.float32
            if self.use_double:
                pass
                # d_type = tf.double

            h = None
            with tf.device(self.device):
                with tf.Graph().as_default():
                    with tf.Session() as sess:
                        px = tf.placeholder(d_type, [None, n_input])
                        py = tf.placeholder(d_type, [None, n_classes])
                        hidden = self.do_transform(px, before_activation=before_activation)
                        init = tf.global_variables_initializer()
                        sess.run(init)

                        data.rewind()
                        while data.has_next_batch():
                            batch_x, batch_y = data.next_batch(batch_size, do_shuffle=False)



                            h_part = sess.run(hidden, feed_dict={px: batch_x, py: batch_y})
                            if h is None:
                                h = np.zeros(shape=(data.num_examples, h_part.shape[1]))
                            si = data.batch_index * batch_size
                            ei = si + h_part.shape[0]
                            h[si:ei, :] = h_part
            return h

    def process_dataset(self, dataset, normalization=""):
        with tf.Graph().as_default():
            with tf.device("/cpu:0"):
                with tf.Session() as sess:
                    x = self.do_transform(dataset.train.images)
                    sess.run(x)
                    dataset.train.images = x.eval()

                    if hasattr(dataset, 'test'):
                        x = self.do_transform(dataset.test.images)
                        sess.run(x)
                        dataset.test.images = x.eval()

                    if hasattr(dataset, 'validation'):
                        x = self.do_transform(dataset.validation.images)
                        sess.run(x)
                        dataset.validation.images = x.eval()
                    sess.close()

            if normalization != "":
                DataInput.normalize(dataset, normalization)
        self.__processed = True

    def predict(self, data, release_after_predict=False):
        if self.sess is None:
            self.setup_tensor_ops()

        data_count = len(data)
        batch_count = int(data_count / self.batch_size)
        remain_count = data_count % self.batch_size
        if remain_count > 0:
            batch_count += 1

        pred = None
        for bi in range(batch_count):

            si = bi * self.batch_size
            if bi == batch_count - 1 and remain_count > 0:
                ei = si + remain_count
            else:
                ei = si + self.batch_size

            feed_dict = {self.inp: data[si:ei]}
            if self.phase_train is not None:
                feed_dict[self.phase_train] = False

            h = self.sess.run(self.out, feed_dict=feed_dict)

            if pred is None:
                pred = np.zeros(shape=(data_count, h.shape[1]))
            pred[si:ei] = h

        if release_after_predict:
            self.sess.close()

        return pred

    def train(self, data, C=0.0, h_out=None):
        # Plain Training
        if h_out is None:
            if self.__w1 is None:
                self.init_weight(data, self.train_param.n_hidden, self.train_param.init_name, self.train_param.init_param,
                                 display_step=self.display_step)
                C = self.train_param.c

            self.__w1 = self.__w1.astype(np.float32)
            self.__b1 = self.__b1.astype(np.float32)

            h = self.do_transform(None, data.train, before_activation=False)

        # Just solve output layer (when hidden-feature is already calculated)
        else:
            h = h_out

        # h_avg = np.mean(h, axis=0)
        # h_std = np.std(h, axis=0)

        if self.activation_normalizer is not None:
            h = self.activation_normalizer.train(h)

        if self.use_tf_solver:
            ret = ELM.solve_linear_tf(h, data.train.labels, C, ret=self.tf_solver_ctx,
                                      gpu_usage_level=ELM.OPTION_TF_SOLVER_LEVEL, device=self.device)
            if self.use_tf_solver_reuse:
                self.tf_solver_ctx = ret
            else:
                ret.sess.close()
            w2 = ret.w

        elif self.use_af_solver:
            w2 = ELM.solve_linear_af(h, data.train.labels)
        else:
            w2 = ELM.solve_linear(h, data.train.labels, C, self.sample_weight, self.use_double,
                                  dispose_x=(not self.store_h))

        self.__w2 = w2
        if self.store_h:
            self.__h = h
        self.__w2 = w2.astype(np.float32)
        self.__b2 = np.full([data.train.dim_labels], 0.0, dtype=np.float32)

    def train_os(self, data, C=0.0, is_update=False, use_double=False):
        if self.__w1 is None:
            if self.train_param is None:
                raise Exception('ELM input weight is not initialized yet. Call init_weight() or set train_param')
            else:
                self.init_weight(data, self.train_param.n_hidden, self.train_param.init_name,
                                 self.train_param.init_param,
                                 display_step=self.display_step)
                C = self.train_param.c

        if is_update and self.__p is None:
            raise Exception('initial training of OS-ELM is not performed yet')

        self.__w1 = self.__w1.astype(np.float32)
        self.__b1 = self.__b1.astype(np.float32)

        h = self.do_transform(None, data.train)

        if not is_update:
            w2, pinv = ELM.solve_linear(h, data.train.labels, C, self.sample_weight, self.use_double,
                                        dispose_x=(not self.store_h), return_pinv=True)
        else:
            if self.use_tf_solver:
                ret = ELM.solve_linear_tf_update(h, data.train.labels, self.__p, self.__w2, ret=self.tf_solver_ctx)
                w2 = ret.b_n
                pinv = ret.p_n
                if self.use_tf_solver_reuse:
                    self.tf_solver_ctx = ret
                else:
                    ret.sess.close()
            elif self.use_af_solver:
                w2, pinv = ELM.solve_linear_af_update(h, data.train.labels, self.__p, self.__w2, use_double)
            else:
                w2, pinv = ELM.solve_linear_update(h, data.train.labels, self.__p, self.__w2)

        self.__w2 = w2
        self.__p = pinv
        self.__w2 = w2.astype(np.float32)
        self.__b2 = np.full([data.train.dim_labels], 0.0, dtype=np.float32)

    def setup_tensor_ops(self, release_cpu_memory=False):
        if self.sess is not None:
            self.sess.close()

        config = tf.ConfigProto(allow_soft_placement=False,
                                log_device_placement=self.device_log)
        with tf.Graph().as_default():
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            with tf.device(self.device):
                self.inp = tf.placeholder(tf.float32, [None, self.__w1.shape[0]])
                self.out = self.build_model(self.inp)

        if release_cpu_memory:
            del self.__w1
            self.__w1 = None
            del self.__w2
            self.__w2 = None
            del self.__b1
            self.__b1 = None
            del self.__b2
            self.__b2 = None

    def train_sgd(self, data, n_hidden, training_epochs, learning_rate=(1e-3),
                  momentum=0.0, weight_decay=(), converge_count=0.0, converge_delta=0.0, display_step=0,
                  w1_tune=False, w1_bn=False, use_softmax=True, use_adam=False, do_initialization=True,
                  do_shuffle_per_epoch=True, use_output_bias=True, init_type='Xavier-Normal'):

        # Parameters
        batch_size = self.batch_size

        # Network Parameters
        n_input = data.train.dim_features
        n_classes = data.train.dim_labels

        best_cost = 1e+10
        best_past_count = 0

        with tf.device(self.device):
            with tf.Graph().as_default():
                self.sess = tf.Session()
                self.phase_train = tf.Variable(True)

                d_type = tf.float32
                x = tf.placeholder(d_type, [None, n_input])
                y = tf.placeholder(d_type, [None, n_classes])

                # 1-Hidden-Layer NN
                if w1_tune:
                    w1 = tf.Variable(self.__w1)
                    b1 = tf.Variable(tf.random_normal([n_hidden], dtype=d_type))
                # ELM style
                else:
                    w1 = self.__w1
                    b1 = self.__b1

                if do_initialization:
                    w2, b2 = ELM.create_init_weight(data, n_hidden, n_classes, init_type)
                    w2 = tf.Variable(w2)
                    if use_output_bias:
                        b2 = tf.Variable(w2[0])
                    else:
                        b2 = np.full([n_classes], 0).astype(np.float32)
                else:
                    w2 = tf.Variable(self.__w2)
                    b2 = tf.Variable(self.__b2)

                pred = self.build_model(x, w1, b1, w2, b2)

                batch = tf.Variable(0)
                total_batch = int(data.train.num_examples / batch_size)

                lr_rate = learning_rate[0]
                lr_decay_epoch = total_batch + 1
                lr_decay_ratio = 1.0

                if len(learning_rate) > 1:
                    lr_decay_epoch = learning_rate[1]
                    lr_decay_ratio = learning_rate[2]

                decay_steps = int(total_batch * lr_decay_epoch * 100)

                # Define loss and optimizer
                exp_learning_rate = tf.train.exponential_decay(
                    lr_rate,  # Base learning rate.
                    batch * batch_size,  # Current index into the dataset.
                    decay_steps,  # Decay step.
                    lr_decay_ratio,  # Decay rate.
                    staircase=True)

                wd_val = 0
                wd_decay_epoch = 0
                wd_decay_ratio = 1.0
                wd_decay_bottom = 0
                if len(weight_decay) > 0:
                    wd_val = weight_decay[0]
                    wd_decay_epoch = weight_decay[1]
                    if wd_decay_epoch <= 0:
                        wd_decay_epoch = training_epochs * 2

                    wd_decay_ratio = weight_decay[2]

                    if len(weight_decay) > 2:
                        wd_decay_bottom = weight_decay[3]

                wd_var = tf.Variable(wd_val, dtype=d_type)
                if use_softmax:
                    train_err = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
                    # train_err = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))
                else:
                    train_err = tf.losses.mean_squared_error(y, pred)
                    # train_err = tf.reduce_mean(tf.pow(pred - y, 2))

                if wd_val > 0:
                    if not self.use_L1_Panelty:
                        regluarization = tf.reduce_mean(tf.multiply(wd_var, tf.nn.l2_loss(w2)))
                    else:
                        regluarization = tf.reduce_mean(tf.multiply(wd_var, tf.abs(w2)))
                else:
                    regluarization = 0

                cost = train_err + regluarization
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    if use_adam:
                        optimizer = tf.train.AdamOptimizer(learning_rate=exp_learning_rate).minimize(cost)
                    else:
                        optimizer = tf.train.MomentumOptimizer(learning_rate=exp_learning_rate, momentum=momentum) \
                            .minimize(cost, global_step=batch)

                # Launch the graph
                # Initializing the variables
                init = tf.global_variables_initializer()
                self.sess.run(init)

                # Training cycle
                data.train.rewind()
                for epoch in range(training_epochs):
                    avg_cost = 0.

                    # Loop over all batches
                    for i in range(total_batch):
                        batch_x, batch_y = data.train.next_batch(batch_size, do_shuffle=do_shuffle_per_epoch)
                        feed_dict = {x: batch_x, y: batch_y, self.phase_train: True}
                        if self.phase_train is not None:
                            feed_dict[self.phase_train] = True

                        _, c = self.sess.run([optimizer, cost], feed_dict=feed_dict)
                        # Compute average loss
                        avg_cost += c / total_batch

                    delta = 0
                    if epoch == 0:
                        best_cost = avg_cost
                        remain_count = converge_count
                    else:
                        delta = best_cost - avg_cost

                        if delta > converge_delta:
                            best_cost = avg_cost
                            remain_count = converge_count

                        else:
                            remain_count -= 1

                        if remain_count <= 0:
                            print('Converged')
                            break


                    # decay of l2_loss weight (optional)
                    if wd_decay_epoch > 0 and epoch > 0 and epoch % wd_decay_epoch == 0 and \
                            wd_val > wd_decay_bottom:
                        wd_val = self.sess.run(wd_var)
                        wd_old_val = wd_val
                        if wd_val > wd_decay_bottom:
                            wd_val *= wd_decay_ratio
                        if wd_val < wd_decay_bottom:
                            wd_val = wd_decay_bottom
                        self.sess.run(tf.assign(wd_var, wd_val))
                        print('\t WD: %e -> %e' % (wd_old_val, wd_val))

                    # Display logs per epoch step
                    if display_step > 0 and epoch % display_step == 0:
                        if delta == 0:
                            print("\t[%d] Cost=%.2e" % (epoch, avg_cost))
                        else:
                            print("\t[%d] Cost=%.2e (d: %.2e / %d)" % (epoch, avg_cost, delta, remain_count))

                if w1_tune:
                    self.__w1 = self.sess.run(w1)
                    self.__b1 = self.sess.run(b1)
                else:
                    self.__w1 = w1
                    self.__b1 = b1

                self.__w2 = self.sess.run(w2)
                if use_output_bias:
                    self.__b2 = self.sess.run(b2)

                if self.use_double:
                    self.__w1 = self.__w1.astype(np.float32)
                    self.__b1 = self.__b1.astype(np.float32)
                    self.__w2 = self.__w2.astype(np.float32)
                    self.__b2 = self.__b2.astype(np.float32)

                self.inp = x
                self.out = pred

    def save(self, filepath):
        if self.__w1 is None or self.__b1 is None or self.__w2 is None or self.__b2 is None:
            raise Exception('Not trained yet!')
        dic = {'w1': self.__w1, 'b1': self.__b1, 'w2': self.__w2, 'b2': self.__b2, 'act': self.activation,
               'P': self.__p}
        with open(filepath, 'wb') as fs:
            pickle.dump(dic, fs)

    def load(self, filepath):
        with open(filepath, 'rb') as fs:
            dic = pickle.load(fs, encoding='bytes')
        self.__w1 = dic['w1']
        self.__b1 = dic['b1']
        self.__w2 = dic['w2']
        self.__b2 = dic['b2']
        self.__p = dic['P']
        self.activation = dic['act']

    def release(self):
        if self.sess is not None:
            self.sess.close()
        del self.__w1
        del self.__b1
        del self.__w2
        del self.__b2

        if self.__h is not None:
            del self.__h


    def do_test(self, data):
        total_sum = 0.0
        batch_count = int(data.num_examples / self.batch_size)
        remain_count = data.num_examples % self.batch_size
        if remain_count != 0:
            batch_count += 1

        for i in range(batch_count):
            batch_size = self.batch_size
            if i == batch_count - 1 and remain_count > 0:
                batch_size = remain_count

            batch_x, batch_y = data.next_batch(batch_size, do_shuffle=False)
            pred = self.predict(batch_x)

            pred_max = np.argmax(pred, axis=1)
            y_max = np.argmax(batch_y, axis=1)
            err_count = np.count_nonzero(pred_max - y_max)
            total_sum += (batch_size - err_count)

        return total_sum / data.num_examples

    def test_dataset(self, dataset):
        train_acc = self.do_test(dataset.train)
        test_acc = self.do_test(dataset.test)
        return [train_acc, test_acc]

    @classmethod
    def create_init_weight(self, data_set, n_fdim, n_hidden, init_method="Normal",
                           init_param=None, train_param=None, w_normalization=None, display_step=1):

        d_type = np.float32
        if train_param is not None and train_param.use_double:
            d_type = np.float64

        g_b1 = np.full([n_hidden], 0, dtype=np.float32)

        def get_valid_subregion(img_width, img_height, w_min, w_max, h_min, h_max, sample=None, v_thresh=0.0):
            while True:
                w = random.randrange(int(img_width * w_min), int(img_height * w_max))
                h = random.randrange(int(img_height * h_min), int(img_height * h_max))
                x = random.randrange(0, img_width - w)
                y = random.randrange(0, img_height - h)

                if x + w >= img_width or y + h >= img_height:
                    continue
                else:
                    '''
                if sample is not None:
                    rf_region = sample[y:y + h, x:x + w, :]
                    v = np.std(rf_region)
                    if v < v_thresh:
                        continue
                '''
                    break
            return (x, y, w, h)

        if init_method == "Normal":
            m = 0.0
            s = 1.0
            if init_param is not None:
                m = init_param[0]
                s = init_param[1]
            g_w1 = np.random.normal(loc=m, scale=s, size=(n_fdim, n_hidden)).astype(d_type)

        elif init_method == "Uniform":
            g_w1 = np.random.uniform(low=init_param[0], high=init_param[1], size=(n_fdim, n_hidden)).astype(d_type)

        elif init_method == "Uniform (0/1)":
            g_w1 = np.random.uniform(low=0, high=1, size=(n_fdim, n_hidden)).astype(d_type)

        elif init_method == "Uniform (-1/1)":
            g_w1 = np.random.uniform(low=-1, high=1, size=(n_fdim, n_hidden)).astype(d_type)

        elif init_method == 'Xavier-Normal':
            stddev = math.sqrt(2.0 / (data_set.train.dim_features + data_set.train.dim_labels + 1))
            g_w1 = np.random.normal(loc=0.0, scale=stddev, size=(n_fdim, n_hidden)).astype(np.float32)

        elif init_method == 'Xavier-Uniform':
            init_range = math.sqrt(6.0 / (data_set.train.dim_features + data_set.train.dim_labels + 1))
            g_w1 = np.random.uniform(low=-init_range, high=init_range, size=(n_fdim, n_hidden)).astype(np.float32)

        elif init_method == "RP":
            g_w1 = np.random.uniform(low=0, high=1, size=(n_fdim, n_hidden)).astype(d_type)
            val = 1 / np.sqrt(3)
            g_w1[g_w1 > 0.5] = val
            g_w1[g_w1 <= 0.5] = -val

        elif init_method == 'ORF':
            if n_fdim > n_hidden:
                max_val = n_fdim
                min_val = n_hidden
            else:
                max_val = n_hidden
                min_val = n_fdim

            g_w1 = np.random.normal(loc=0.0, scale=1, size=(max_val, max_val)).astype(d_type)
            q, r = np.linalg.qr(g_w1)
            s = np.identity(q.shape[0])
            s_m = np.random.normal(loc=0.0, scale=1, size=(1, max_val)).astype(d_type)
            s *= s_m
            g_w1 = np.matmul(s, q)
            g_w1 /= init_param[0]
            g_w1 = g_w1[0:min_val]

        elif init_method == "AE":
            elm_ae = ELM()
            data_set_labels = data_set.train.labels
            data_set.train.labels = data_set.train.images

            elm_ae.init_weight(data_set, n_hidden, train_param=train_param)
            elm_ae.train(data_set)

            data_set.train.labels = data_set_labels
            g_w1 = np.copy(elm_ae.__w2)
            g_w1 = np.transpose(g_w1)

            if False:
                for i in range(n_fdim):
                    g_w1[:, i] /= np.linalg.norm(g_w1[:, i], 2)

        elif init_method == "Binary":
            g_w1 = np.random.uniform(low=0, high=1, size=(n_fdim, n_hidden)).astype(d_type)
            g_w1[g_w1 > 0.5] = 1
            g_w1[g_w1 <= 0.5] = -1

        elif init_method == "Ternary":
            g_w1 = np.random.uniform(low=0, high=1, size=(n_fdim, n_hidden)).astype(d_type)
            g_w1[g_w1 > 0.66] = 1
            g_w1[1 > g_w1 > 0.33] = -0
            g_w1[0.33 > g_w1] = -1

        elif init_method == "RF" or init_method == 'RF-RP':
            g_w1 = np.zeros([n_fdim, n_hidden]).astype(d_type)

            w_min = init_param[0]
            w_max = init_param[1]
            h_min = init_param[2]
            h_max = init_param[3]
            v_thresh = init_param[4]

            images = data_set.train.images.reshape(data_set.train.num_examples,
                                                   data_set.train.width, data_set.train.height, data_set.train.channel)

            for hi in range(n_hidden):
                idx = random.randrange(0, data_set.train.num_examples)
                x, y, w, h = get_valid_subregion(data_set.train.width, data_set.train.height, w_min, w_max, h_min,
                                                 h_max,
                                                 images[idx], v_thresh)

                rf_weight = np.zeros([data_set.train.height, data_set.train.width], dtype=d_type)
                w0 = np.random.uniform(low=0.0, high=1, size=(h, w))

                if init_method == 'RF-RP':
                    w0[w0 > 0.5] = 1
                    w0[w0 <= 0.5] = -1

                rf_weight[y:y + h, x:x + w] = w0
                rf_weight = rf_weight.reshape(1, data_set.train.height * data_set.train.width)
                g_w1[:, hi] = rf_weight

        elif init_method == "RF-AE" or init_method == 'RF-AE-RP':
            g_w1 = np.zeros([n_fdim, n_hidden]).astype(d_type)

            w_min = init_param[0]
            w_max = init_param[1]
            h_min = init_param[2]
            h_max = init_param[3]
            v_thresh = init_param[4]
            h_num = init_param[5]

            h_count = int(n_hidden / h_num)

            if n_hidden % h_num != 0:
                raise Exception('%d / %d is not exactly divisible!' % (n_hidden, h_num))

            images = data_set.train.images.reshape(data_set.train.num_examples,
                                                   data_set.train.width, data_set.train.height, data_set.train.channel)
            for i in range(h_count):
                if display_step > -0 and (i % display_step) == 0:
                    print('\t RF-AE (%d/%d)' % (i + 1, h_count))

                idx = random.randrange(0, data_set.train.num_examples)
                x, y, w, h = get_valid_subregion(data_set.train.width, data_set.train.height, w_min, w_max, h_min,
                                                 h_max,
                                                 images[idx], v_thresh)

                rf_images = images[:, x:x + w, y:y + h, :]
                rf_images = rf_images.reshape(data_set.train.num_examples, w * h * data_set.train.channel)

                sub_data_set = DataInput.DataSets()
                sub_data_set.train = DataInput.DataSet(rf_images, rf_images)

                e = ELM()

                if init_method == 'RF-AE-RP':
                    e.init_weight(sub_data_set, h_num, 'RP')
                else:
                    e.init_weight(sub_data_set, h_num, 'Normal')

                e.train(sub_data_set)

                rf_ae = e.__w2
                rf_ae = rf_ae.reshape(h_num, w, h)
                rf_weight = np.zeros([h_num, data_set.train.height, data_set.train.width], dtype=d_type)
                rf_weight[:, x:x + w, y:y + h] = rf_ae
                rf_weight = rf_weight.reshape(h_num, data_set.train.height * data_set.train.width)

                si = h_num * i
                ei = si + h_num
                g_w1[:, si:ei] = np.transpose(rf_weight)

        elif init_method == "S":
            g_w1 = np.zeros(shape=[n_fdim, n_hidden]).astype(d_type)
            # self.__b1 = np.random.normal(loc=0.0, scale=1/np.sqrt(n_fdim), size=(n_hidden)).astype(d_type)

            idx = np.random.choice(data_set.train.num_examples, n_hidden, replace=False)
            for i in range(len(idx)):
                sample = data_set.train.images[idx[i]].copy()
                sample /= np.linalg.norm(sample, 1)
                g_w1[:, i] = sample.copy()

        elif init_method == 'RF-S':
            g_w1 = np.zeros([n_fdim, n_hidden]).astype(d_type)

            w_min = init_param[0]
            w_max = init_param[1]
            h_min = init_param[2]
            h_max = init_param[3]
            v_thresh = init_param[4]

            images = data_set.train.images.reshape(data_set.train.num_examples,
                                                   data_set.train.width, data_set.train.height, data_set.train.channel)

            idx = np.random.random_integers(0, data_set.train.num_examples - 1, n_hidden)
            for i in range(len(idx)):
                idx = random.randrange(0, data_set.train.num_examples)
                x, y, w, h = get_valid_subregion(data_set.train.width, data_set.train.height, w_min, w_max, h_min,
                                                 h_max,
                                                 images[idx], v_thresh)

                rf_weight = np.zeros([1, data_set.train.height, data_set.train.width, data_set.train.channel],
                                     dtype=d_type)
                rf_weight[0, y:y + h, x:x + w, :] = images[idx, y:y + h, x:x + w, :]
                rf_weight = rf_weight.reshape(1, data_set.train.height * data_set.train.width * data_set.train.channel)
                g_w1[:, i] = np.transpose(rf_weight[0])

        elif init_method == 'CD':
            g_w1 = np.zeros(shape=[n_fdim, n_hidden]).astype(d_type)

            label_idx = np.argmax(data_set.train.labels, axis=1)
            for i in range(n_hidden):
                idx1 = random.randrange(0, data_set.train.num_examples)
                idx2 = random.randrange(0, data_set.train.num_examples)
                diff = data_set.train.images[idx2] - data_set.train.images[idx1]
                zCount = np.count_nonzero(diff)

                while idx1 == idx2 or label_idx[idx1] == label_idx[idx2] or zCount == 0:
                    idx1 = random.randrange(0, data_set.train.num_examples)
                    idx2 = random.randrange(0, data_set.train.num_examples)
                    diff = data_set.train.images[idx2] - data_set.train.images[idx1]
                    zCount = np.count_nonzero(diff)

                diff = data_set.train.images[idx2] - data_set.train.images[idx1]
                diff_n = np.linalg.norm(diff, 2)
                sum = data_set.train.images[idx2] + data_set.train.images[idx1]
                b = np.matmul(np.transpose(sum), sum) / diff_n

                g_w1[:, i] = (2 * diff) / diff_n
                g_b1[i] = b

        elif init_method == 'RF-CD':
            g_w1 = np.zeros([n_fdim, n_hidden]).astype(d_type)

            w_min = init_param[0]
            w_max = init_param[1]
            h_min = init_param[2]
            h_max = init_param[3]
            v_thresh = init_param[4]

            label_idx = np.argmax(data_set.train.labels, axis=1)

            for i in range(n_hidden):

                idx1 = random.randrange(0, data_set.train.num_examples)
                idx2 = random.randrange(0, data_set.train.num_examples)
                diff = data_set.train.images[idx2] - data_set.train.images[idx1]
                zCount = np.count_nonzero(diff)

                while idx1 == idx2 or label_idx[idx1] == label_idx[idx2] or zCount == 0:
                    idx1 = random.randrange(0, data_set.train.num_examples)
                    idx2 = random.randrange(0, data_set.train.num_examples)
                    diff = data_set.train.images[idx2] - data_set.train.images[idx1]
                    zCount = np.count_nonzero(diff)

                diff = diff.reshape(data_set.train.width, data_set.train.height, data_set.train.channel)
                x, y, w, h = get_valid_subregion(data_set.train.width, data_set.train.height, w_min, w_max, h_min,
                                                 h_max,
                                                 diff, v_thresh)

                diff_n = diff[y:y + h, x:x + w, :].reshape(1, h * w * data_set.train.channel)
                diff_n = np.linalg.norm(diff_n, 2)

                if math.isnan(diff_n) or diff_n == 0:
                    i -= 1
                    continue

                sum = data_set.train.images[idx2] + data_set.train.images[idx1]
                sum = sum.reshape(1, data_set.train.width, data_set.train.height, data_set.train.channel)
                sum = np.sum(np.power(sum[0, y:y + h, x:x + w, :], 2))
                b = sum / diff_n

                rf_weight = np.zeros([data_set.train.height, data_set.train.width, data_set.train.channel],
                                     dtype=d_type)
                rf_weight[y:y + h, x:x + w, :] = (2 * diff[y:y + h, x:x + w, :]) / diff_n
                rf_weight = rf_weight.reshape(1, data_set.train.height * data_set.train.width * data_set.train.channel)

                g_w1[:, i] = np.transpose(rf_weight[0])
                g_b1[i] = b

        if w_normalization is not None:
            if w_normalization[0] == 'Std':
                e = ELM()

                e.activation = train_param.activation
                e.use_double = train_param.use_double
                e.__w1 = g_w1.astype(np.float32)
                e.__b1 = g_b1.astype(np.float32)

                h = e.do_transform(None, data_set.train, before_activation=True)
                h_avg = np.average(h, axis=0) + w_normalization[1]
                h_std = np.std(h, axis=0) / w_normalization[2]

                for i in range(n_hidden):
                    if h_std[i] != 0:
                        g_w1[:, i] = g_w1[:, i] / h_std[i]
                        g_b1[i] = -(h_avg[i] / h_std[i])

                if not e.use_double:
                    g_w1 = g_w1.astype(np.float32)
                    g_b1 = g_b1.astype(np.float32)

                e.__w1 = g_w1
                e.__b1 = g_b1

                # h = e.do_transform(None, data_set.train, before_activation=True)
                # h_avg = np.average(h, axis=0)
                # h_std = np.std(h, axis=0)

            elif w_normalization[0] == 'Sum':
                g_sum = np.sum(g_w1, axis=0)
                g_w1 /= g_sum

            else:
                raise Exception('Invalid w_normalization = %s!' % w_normalization[0])

        return (g_w1, g_b1)

    @classmethod
    def solve_linear(self, x, y, C=0, weight=None, use_double=False, dispose_x=False, return_pinv=False):
        n_input = np.shape(x)[0]
        n_fdim = np.shape(x)[1]
        pinv = None

        if not use_double:
            x = x.astype(np.float32)
        x_t = np.transpose(x)

        if n_input > n_fdim:
            if weight is not None:
                x_sqr = np.matmul(x_t, weight)
                x_sqr = np.matmul(x_sqr, x)
            else:
                x_sqr = np.matmul(x_t, x)

            if C > 0:
                c_mat = np.identity(n_fdim) * C
                x_sqr += c_mat

            if dispose_x:
                x = None

            if return_pinv:
                pinv = x_sqr

            x_sqr = np.linalg.pinv(x_sqr)
            w = np.matmul(x_sqr, x_t)

        else:
            x_sqr = np.matmul(x, x_t)
            if weight is not None:
                x_sqr = np.matmul(weight, x_sqr)

            if C > 0:
                c_mat = np.identity(n_input) * C
                x_sqr += c_mat

            if dispose_x:
                x = None

            if return_pinv:
                pinv = x_sqr

            x_sqr = np.linalg.pinv(x_sqr)
            w = np.matmul(x_t, x_sqr)

        ny = y
        w = np.matmul(w, ny)

        if return_pinv:
            return [w, pinv]
        else:
            return w

    @classmethod
    def solve_linear_update(cls, x, y, p, b):
        x_t = np.transpose(x)
        px_t = np.matmul(p, x_t)

        eq_1 = np.matmul(x, px_t)
        eq_1 = eq_1 + np.eye(eq_1.shape[0], eq_1.shape[1], dtype=eq_1.dtype)
        eq_1 = np.linalg.pinv(eq_1)
        eq_1 = np.matmul(px_t, eq_1)
        eq_1 = np.matmul(eq_1, x)
        eq_1 = np.matmul(eq_1, p)
        p_n = p - eq_1

        eq_2 = np.matmul(x, b)
        eq_2 = (y - eq_2)
        eq_3 = np.matmul(p_n, x_t)

        if eq_2.shape[1] == b.shape[1]:
            b_n = b + np.matmul(eq_3, eq_2)
        else:
            b_n = b + np.matmul(eq_2, eq_3)
        return b_n, p_n

    @classmethod
    def solve_linear_sgd(cls, x, y, n_epoch, learning_rate, momentum=0, learning_decay=None,
                         weight_decay=None, stop_delta=0, display_step=0, use_softmax=False, use_double=False):

        batch_size = 100
        cost_list_max = 50

        batch = tf.Variable(0)
        total_batch = int(x.shape[0] / batch_size)

        lr_decay_epoch = total_batch + 1
        lr_decay_ratio = 0

        if learning_decay is not None:
            lr_decay_epoch = learning_decay[0]
            lr_decay_ratio = learning_decay[1]

        decay_steps = int(total_batch * lr_decay_epoch * 100)

        # Define loss and optimizer
        exp_learning_rate = tf.train.exponential_decay(
            learning_rate,  # Base learning rate.
            batch * batch_size,  # Current index into the dataset.
            decay_steps,  # Decay step.
            lr_decay_ratio,  # Decay rate.
            staircase=True)

        wd_val = 0
        wd_decay_epoch = 0
        wd_decay_ratio = 1.0
        wd_decay_bottom = 0
        if weight_decay is not None:
            wd_val = weight_decay[0]
            wd_decay_epoch = weight_decay[1]
            wd_decay_ratio = weight_decay[2]
            wd_decay_bottom = weight_decay[3]

        if not use_double:
            bx = tf.placeholder(tf.float32, [None, x.shape[1]])
            by = tf.placeholder(tf.float32, [None, y.shape[1]])
            w = tf.Variable(tf.random_normal([x.shape[1], y.shape[1]], dtype=tf.float32))
            wd_var = tf.Variable(wd_val, dtype=tf.float32)
        else:
            bx = tf.placeholder(tf.float64, [None, x.shape[1]])
            by = tf.placeholder(tf.float64, [None, y.shape[1]])
            w = tf.Variable(tf.random_normal([x.shape[1], y.shape[1]], dtype=tf.float64))
            wd_var = tf.Variable(wd_val, dtype=tf.float64)

        pred = tf.matmul(bx, w)
        if wd_val > 0:
            if use_softmax:
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=by)) + \
                       tf.multiply(wd_var, tf.nn.l2_loss(w))
            else:
                cost = tf.reduce_mean(tf.pow(pred - by, 2)) + tf.multiply(wd_var, tf.nn.l2_loss(w))
        else:
            if use_softmax:
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=by))
            else:
                cost = tf.reduce_mean(tf.pow(pred - by, 2))

        optimizer = tf.train.MomentumOptimizer(learning_rate=exp_learning_rate, momentum=momentum).minimize(cost,
                                                                                                            global_step=batch)

        cost_list = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Training cycle
            for epoch in range(n_epoch):
                avg_cost = 0.

                for bi in range(total_batch):
                    si = bi * batch_size
                    ei = si + batch_size

                    batch_x = x[si:ei]
                    batch_y = y[si:ei]
                    _, c = sess.run([optimizer, cost], feed_dict={bx: batch_x, by: batch_y})
                    avg_cost += c / total_batch

                cost_list.append(avg_cost)
                delta = 0
                if cost_list_max < len(cost_list):
                    delta = cost_list[0] - cost_list[cost_list_max]
                    if avg_cost == 0 or 0 <= delta < stop_delta:
                        print("\t[%d] Cost=%.2e (d: %.2e)" % (epoch, avg_cost, delta))
                        print("\tConverged")
                        break
                    cost_list.pop(0)

                if wd_decay_epoch > 0 and epoch > 0 and epoch % wd_decay_epoch == 0 and \
                        wd_val > wd_decay_bottom:
                    wd_val = wd_var.eval()
                    wd_old_val = wd_val
                    if wd_val > wd_decay_bottom:
                        wd_val *= wd_decay_ratio
                    if wd_val < wd_decay_bottom:
                        wd_val = wd_decay_bottom
                    sess.run(tf.assign(wd_var, wd_val))
                    print('\t WD: %e -> %e' % (wd_old_val, wd_val))

                if display_step > 0 and epoch % display_step == 0:
                    if delta == 0:
                        print("\t[%d] Cost=%.2e" % (epoch, avg_cost))
                    else:
                        print("\t[%d] Cost=%.2e (d: %.2e)" % (epoch, avg_cost, delta))

            w = w.eval()
        if use_double:
            w = w.astype(np.float32)
        return w

    @classmethod
    def solve_scipy(cls, x, y):

        def fit_f(param, data):
            pred = np.matmul(data, param)
            return pred

        def err_f(param, data, target):
            pred = fit_f(param, data)
            err = target - pred
            return err

        n_fdim = x.shape[1]
        n_tdim = y.shape[1]

        w = np.random.normal(loc=0, scale=1, size=(n_fdim, n_tdim))
        for li in range(n_tdim):
            result = optimize.least_squares(err_f, w[:, li], args=(x, y[:, li]))
            w[:, li] = result.x
            print('[%d] %.4f' % (li, result.cost))
        return w

    @classmethod
    def solve_linear_tf(cls, x, y, C=0, ret=None, gpu_usage_level=1, device='/gpu:0'):
        n_input = np.shape(x)[0]
        n_fdim = np.shape(x)[1]

        # x = x.astype(np.float32)

        # Only solve inverse operation on the GPU
        if gpu_usage_level == 0:
            x = x.astype(np.float64)
            y = y.astype(np.float64)

            if n_input > n_fdim:
                x_sqr = np.matmul(np.transpose(x), x)
                if C > 0:
                    c_mat = np.identity(n_fdim) * C
                    x_sqr += c_mat
            else:
                x_sqr = np.matmul(x, np.transpose(x))
                if C > 0:
                    c_mat = np.identity(n_input) * C
                    x_sqr += c_mat

            if ret is None:
                ret = util.Result()
                ret.graph = tf.Graph()
                ret.sess = tf.Session()
                ret.graph.as_default()
                with tf.device(device):
                    ret.x_sqr = tf.placeholder(dtype=tf.float64, shape=x_sqr.shape)
                    ret.x_inv = tf.matrix_inverse(ret.x_sqr)

            x_inv = ret.sess.run(ret.x_inv, feed_dict={ret.x_sqr: x_sqr})

            if n_input > n_fdim:
                w = np.matmul(x_inv, np.transpose(x))
            else:
                w = np.matmul(np.transpose(x), x_inv)

            w = np.matmul(w, y)
            ret.w = w.astype(np.float32)
            return ret

        # Solve matrix multiplication and inverse on the GPU
        else:
            x = x.astype(np.float64)
            y = y.astype(np.float64)

            if ret is None:
                ret = util.Result()
                ret.graph = tf.Graph()
                ret.sess = tf.Session()
                ret.graph.as_default()
                with tf.device(device):
                    ret.x = tf.placeholder(dtype=tf.float64, shape=x.shape)
                    ret.y = tf.placeholder(dtype=tf.float64, shape=y.shape)

                    if n_input > n_fdim:
                        x_sqr = tf.matmul(ret.x, ret.x, transpose_a=True)
                        if C > 0:
                            c_mat = np.identity(n_fdim) * C
                            x_sqr += c_mat
                    else:
                        x_sqr = tf.matmul(ret.x, ret.x, transpose_b=True)
                        if C > 0:
                            c_mat = np.identity(n_input) * C
                            x_sqr += c_mat
                    x_inv = tf.matrix_inverse(x_sqr)

                    if n_input > n_fdim:
                        w = tf.matmul(x_inv, ret.x, transpose_b=True)
                    else:
                        w = tf.matmul(ret.x, x_inv, transpose_a=True)
                    ret.sol = tf.matmul(w, ret.y)

            w = ret.sess.run(ret.sol, feed_dict={ret.x:x, ret.y:y})
            ret.w = w.astype(np.float32)
            return ret

    @classmethod
    def solve_linear_tf_update(cls, x, y, p, b, ret=None, device='/gpu:0'):
        x = x.astype(np.float64)
        y = y.astype(np.float64)

        if ret is None:
            ret = util.Result()
            ret.graph = tf.Graph()
            ret.sess = tf.Session()
            ret.graph.as_default()

            with tf.device(device):

                ret.x = tf.placeholder(dtype=tf.float64, shape=x.shape)
                ret.y = tf.placeholder(dtype=tf.float64, shape=y.shape)
                ret.p = tf.placeholder(dtype=tf.float64, shape=p.shape)
                ret.b = tf.placeholder(dtype=tf.float64, shape=b.shape)

                x_t = tf.transpose(ret.x)
                px_t = tf.matmul(ret.p, x_t)
                eq_1 = tf.matmul(ret.x, px_t)
                eq_1_shape = eq_1.shape
                eq_1 = eq_1 + np.eye(eq_1_shape[0], eq_1_shape[1])
                eq_1 = tf.matrix_inverse(eq_1)
                eq_1 = tf.matmul(px_t, eq_1)
                eq_1 = tf.matmul(eq_1, ret.x)
                eq_1 = tf.matmul(eq_1, ret.p)
                p_n = ret.p - eq_1

                eq_2 = tf.matmul(ret.x, ret.b)
                eq_2 = (ret.y - eq_2)
                eq_3 = tf.matmul(p_n, x_t)

                if eq_2.shape[1] == b.shape[1]:
                    b_n = b + tf.matmul(eq_3, eq_2)
                else:
                    b_n = b + tf.matmul(eq_2, eq_3)

                ret.sol1 = b_n
                ret.sol2 = p_n

        b_n, p_n = ret.sess.run([ret.sol1, ret.sol2],
                                feed_dict={ret.x: x, ret.y: y, ret.p: p, ret.b: b})
        ret.b_n = b_n
        ret.p_n = p_n
        return ret

    @classmethod
    def solve_linear_af(cls, x, y, backend='cuda'):
        '''
        x_arr = af.np_to_af_array(x)
        y_arr = af.np_to_af_array(y)

        w = af.solve(x_arr, y_arr)
        w = w.__array__()
        return w
        '''
        raise Exception('Deprecated option')

    @classmethod
    def solve_linear_af_update(cls, x_a, y_a, p_a, b_a, use_double=False):
        '''
        d_type = np.float32
        if use_double:
            d_type = np.double

        x = af.np_to_af_array(x_a.astype(d_type))
        y = af.np_to_af_array(y_a.astype(d_type))
        p = af.np_to_af_array(p_a.astype(d_type))
        b = af.np_to_af_array(b_a.astype(d_type))

        x_t = af.transpose(x)
        px_t = af.matmul(p, x_t)

        eq_1 = af.matmul(x, px_t)
        eq_1 = eq_1 + af.identity(eq_1.dims()[0], eq_1.dims()[1])
        eq_1 = af.inverse(eq_1)

        eq_1 = af.matmul(px_t, eq_1)
        eq_1 = af.matmul(eq_1, x)
        eq_1 = af.matmul(eq_1, p)
        p_n = p - eq_1

        eq_2 = af.matmul(x, b)
        eq_2 = (y - eq_2)
        eq_3 = af.matmul(p_n, x_t)

        if eq_2.dims()[1] == b.dims()[1]:
            b_n = b + af.matmul(eq_3, eq_2)
        else:
            b_n = b + af.matmul(eq_2, eq_3)

        p_n = p_n.__array__()
        b_n = b_n.__array__()
        return b_n, p_n
        '''
        raise Exception('Deprecated option')


def get_tensorflow_layers(input, elm):
    in_w = tf.Variable(elm.input_weight)
    in_b = tf.Variable(elm.input_bias)

    out_w = tf.Variable(elm.output_weight)
    out_b = tf.Variable(elm.output_bias)

    fc1 = tf.matmul(input, in_w) + in_b
    if elm.activation == 'Sigmoid':
        act = tf.sigmoid(fc1)
    elif elm.activation == 'ReLU':
        act = tf.nn.elu(fc1)
    elif elm.activation == 'ExpLU':
        layer1 = tf.nn.elu(fc1)
    elif elm.activation == 'LeakyReLU':
        act = tf.nn.leaky_relu(fc1, 0.1)
    else:
        raise Exception('Unsupported activation! = %s' % elm.activation)

    fc2 = tf.matmul(act, out_w) + out_b
    return [fc1, act, fc2]


def get_CNN_filters(elm, filter_size):
    in_w = np.transpose(elm.input_weight)
    out_w = np.transpose(elm.output_weight)

    num_hidden = in_w.shape[0]
    dim_hidden = in_w.shape[1]
    dim_filter = filter_size[0] * filter_size[1] * filter_size[2]
    dim_label = out_w.shape[0]
    num_filter = filter_size[3]

    if dim_hidden != dim_filter:
        raise Exception('Input dimension and filter size mismatch! (%d != %d)' % (dim_hidden, dim_filter))

    if dim_label != num_filter:
        raise Exception('Output dimension and filter number mismatch! (%d != %d)' % (dim_label, num_filter))

    # Reshape and change order
    in_w = np.reshape(in_w, newshape=(num_hidden, filter_size[0], filter_size[1], filter_size[2]))
    in_w = np.transpose(in_w, (1, 2, 3, 0))

    out_w = np.reshape(out_w, newshape=(dim_label, 1, 1, num_hidden))
    out_w = np.transpose(out_w, (1, 2, 3, 0))

    return in_w, out_w


def get_class_balanced_sample_weight(data) :
    weight_mat = np.zeros(shape=(data.train.num_examples, data.train.num_examples), dtype=np.float32)
    n_class_samples = np.sum(data.train.labels, axis=0)
    label_idx = np.argmax(data.train.labels, axis=1)

    for ri in range(data.train.num_examples):
        n_cls = n_class_samples[label_idx[ri]]
        weight_mat[ri][ri] = 1.0 / n_cls

    return weight_mat


def get_error_balanced_sample_weight(data, n_hidden):
    elm = ELM()
    elm.init_weight(data, n_hidden, 'Xavier-Normal')
    elm.train(data)
    pred = elm.predict(data.train.images)

    label_idx = np.argmax(data.train.labels, axis=1)
    weight_mat = np.zeros(shape=(data.train.num_examples, data.train.num_examples), dtype=np.float32)
    for ri in range(data.train.num_examples):
        idx = label_idx[ri]
        diff = data.train.labels[ri][idx] - pred[ri][idx]
        if diff < 0 :
            diff = 0
        weight_mat[ri][ri] = diff

    return weight_mat


def discriminateive_clustering(data, init_label, n_hidden, batch_size, max_iter, class_max_ratio=1.0):
    n_init = init_label.shape[0]
    n_labels = init_label.shape[1]

    e = ELM()
    e.init_weight(data, n_hidden, 'RP')

    # Initial Training
    b_images = data.train.images[:n_init]
    b_labels = init_label[:]
    sub_data = DataInput.DataSets.create(b_images, b_labels)
    e.train_os(sub_data, is_update=False)

    data.train.rewind()
    while data.train.has_next_batch():
        b_images, b_labels = data.train.next_batch(batch_size, do_shuffle=False)
        p_labels = e.predict(b_images)
        p_labels = np.argmax(p_labels, axis=1)
        p_labels = DataInput.dense_to_one_hot(p_labels, n_labels)
        sub_data = DataInput.DataSets.create(b_images, p_labels)
        e.train_os(sub_data, is_update=True)

    prev_labels = None
    cls_sample_num_max = int((data.train.num_examples / n_labels) * class_max_ratio)
    for iter in range(max_iter):
        cur_labels = np.zeros(shape=(data.train.num_examples, 1), dtype=np.uint32)
        class_count = np.full(shape=(data.train.num_examples, 1), fill_value=cls_sample_num_max)
        constraint_count = data.train.num_examples - (data.train.num_examples % n_labels)
        pred = e.predict(data.train.images)

        # Pseudo label assignment with class-balance constraint
        for sample_idx in range(data.train.num_examples):
            max_list = np.argsort(pred[sample_idx])[:-1]

            for max_idx in max_list:
                # Force class balance constraint
                if sample_idx < constraint_count:
                    if class_count[max_idx] > 0:
                        cur_labels[sample_idx] = max_idx
                        class_count[max_idx] -= 1
                        break
                # Assign maximum value without balance constraint
                else:
                    cur_labels[sample_idx] = max_idx
                    break

        # Convergence Check
        if prev_labels is not None:
            if np.count_nonzero(prev_labels - cur_labels) == 0:
                break
        prev_labels = cur_labels

        # Re-training
        if iter < max_iter - 1:
            batch_count = data.train.num_examples // batch_size
            batch_remain = data.train.num_examples % batch_size
            data.train.rewind()
            for bi in range(batch_count):
                si = bi * batch_size
                ei = si + batch_size
                if bi == batch_count - 1:
                    ei += batch_remain
                b_images = data.train.images[si:ei]
                p_labels = DataInput.dense_to_one_hot(cur_labels[si:ei], n_labels)
                sub_data = DataInput.DataSets.create(b_images, p_labels)
                e.train_os(sub_data, is_update=True)

    return prev_labels


def create_batch_norm(x, phase_train, init_mean=None, init_var=None, var_list=None, name=''):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    Impl_Type = 'Contrib'

    # Use tf.nn.batch_normalization
    if Impl_Type == 'Standard':
        with tf.variable_scope('bn'):
            n_out = x.shape[-1]
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)

            if init_mean is not None and init_var is not None:
                batch_mean = init_mean
                batch_var = init_var
            else:
                # For Conv Layer
                if len(x.shape) == 3:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                # For FC Layer
                else:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3, name=name)

        if var_list is not None:
            var_list.append(beta)
            var_list.append(gamma)

    # Use tf.contrib.layers.batch_norm
    else:
        normed = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase_train)

    return normed
