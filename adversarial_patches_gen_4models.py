from __future__ import division, print_function, absolute_import
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import imageio
import cv2
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib import slim
from lib.core.model.facebox.net import facebox_backbone




from train_config import config as cfg

from get_prob_tensor_dsfd import output_dsfd_predection


from get_prob_tensor_facebox import output_facebox_predection
import mtcnn.mtcnn as mtcnn
import utils.inter_area as inter_area
import utils.patch_mng as patch_mng

from get_prob_tensor_v2 import output_baidu_predection


import os
import shutil
import time

grad_test = 0
mask_value = 0
# ===================================================
# Define class for training procedure
# ===================================================
reader=pywrap_tensorflow.NewCheckpointReader('model/epoch_181L2_0.0005.ckpt')
var_to_shape_map=reader.get_variable_to_shape_map()
reader2=pywrap_tensorflow.NewCheckpointReader('pyramidbox/model/pyramidbox.ckpt')
var_to_shape_map2=reader2.get_variable_to_shape_map()
pyramid_var1 = []
param1 =[]
pyramid_var2 = []
param2 =[]
for key in var_to_shape_map:    
#    print ("tensor_name",key) 
    pyramid_var1.append(key)
    param1.append(reader.get_tensor(key))

for key in var_to_shape_map2:    
#    print ("tensor_name",key) 
    pyramid_var2.append(key)
    param2.append(reader2.get_tensor(key))
    
reader3=pywrap_tensorflow.NewCheckpointReader('dsfd/model/epoch_59L2_0.00025.ckpt')
var_to_shape_map3=reader3.get_variable_to_shape_map()
pyramid_var3 = []
param3 =[]
for key in var_to_shape_map3:    
#    print ("tensor_name",key) 
    pyramid_var3.append(key)
    param3.append(reader3.get_tensor(key))

class TrainMask:
    def __init__(self, gpu_id=4):
        self.pm = patch_mng.PatchManager()
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
        self.compiled = 0
        self.masks_tf = []
        self.sizes = []
        self.bb = tf.Variable(initial_value=1, dtype=tf.float32)
        self.eps = tf.Variable(initial_value=0, dtype=tf.float32)
        self.mu = tf.Variable(initial_value=0, dtype=tf.float32)
        self.L2_reg_fcbox = tf.constant(0.00001)
        self.L2_reg_dsfd = tf.constant(5.e-4)
        self.accumulators = []

    # ===================================================
    # All masks should be within 0..255 range otherwise will be clipped
    # Color: RGB
    # NOTE: the mask itself can be either b/w or color (HxWx1 or HxWx3)
    # ===================================================
    def add_masks(self, masks):
        for key in masks.keys():
            data = masks[key]
            mask = self.pm.add_patch(data[0].clip(0, 255),
                                     key, data[1][::-1], data[2][::-1])
            self.masks_tf.append(mask)

    # ===================================================
    # All images should be located in 'input_img' directory
    # ===================================================
    def add_images(self, images):
        for filename in images:
            img = cv2.imread("input_img/" + filename, cv2.IMREAD_COLOR)
            self.pm.add_image(img)

    # ===================================================
    # Here all TF variables will be prepared
    # The method could be re-run to restore initial values
    # ===================================================
    def build(self, sess):
        if (self.compiled == 0):
            self.pm.compile()
            self.init = self.pm.prepare_imgs()
            self.init_vars = self.pm.init_vars()
            for i, key in enumerate(self.pm.patches.keys()):
                mask_tf = self.pm.patches[key].mask_tf
                accumulator = tf.Variable(tf.zeros_like(mask_tf))
                self.accumulators.append(accumulator)
            self.init_accumulators = tf.initializers.variables(self.accumulators)
            self.compiled = 1

        sess.run(self.init_vars)
        sess.run(self.init)
        sess.run(self.init_accumulators)

    # ===================================================
    # Set the sizes pictures will be scaled
    # ===================================================
    def set_input_sizes(self, sizes):
        self.sizes = sizes

    # ===================================================
    # Here the batch of images will be resized, transposed and normalized
    # ===================================================
    def scale(self, imgs, h, w):
        scaled = inter_area.resize_area_batch(tf.cast(imgs, tf.float64), h, w)
        transposed = tf.transpose(tf.cast(scaled, tf.float32), (0, 2, 1, 3))
        normalized = ((transposed * 255) - 127.5) * 0.0078125
        return normalized

    # ===================================================
    # Build up training function to be used for attacking
    # ===================================================
    def build_train(self, sess, config):
        size2str = (lambda size: str(size[0]) + "x" + str(size[1]))
        pnet_loss = []
        patch_loss = []
        eps = self.eps
        mu = self.mu
        bb = self.bb#无意义，单纯为了使得apply_net_loss可以执行而使用
        L2_reg = self.L2_reg_fcbox #无意义，测试facebox_backbone程序
        L2_reg_dsfd = self.L2_reg_dsfd
        mask_assign_op = []
        moment_assign_op = []
        grad_tf_op = []
        cla_test_op =[]
        facebox_loss_total = []
        loss_bd = []
        loss_bd_pic = []
        dsfd_loss = []

        # Apply all patches and augment
        img_w_mask = self.pm.apply_patches(config.colorizer_wb2rgb)
        self.img_hat = img_w_mask
        noise = tf.random_normal(shape=tf.shape(img_w_mask), mean=0.0, stddev=0.02, dtype=tf.float32)
        img_w_mask = tf.clip_by_value(img_w_mask + noise, 0.0, 1.0)

#        # Create PNet for each size and calc PNet probability map loss
        for size in self.sizes:
            img_scaled = self.scale(img_w_mask, size[0], size[1])
            with tf.variable_scope('pnet_' + size2str(size), reuse=tf.AUTO_REUSE):
                pnet = mtcnn.PNet({'data': img_scaled}, trainable=False)
                pnet.load(os.path.join("./weights", 'det1.npy'), sess)
                clf = sess.graph.get_tensor_by_name("pnet_" + size2str(size) + "/prob1:0")
                bb = sess.graph.get_tensor_by_name("pnet_" + size2str(size) + "/conv4-2/BiasAdd:0")
                pnet_loss.append(config.apply_pnet_loss(clf, bb))
        pnet_loss_total = tf.add_n(pnet_loss)
        tf.summary.scalar('pnet_loss_total',pnet_loss_total)
        
        
        #######################test0910forfacebox###############################
        img_w_mask_255 = img_w_mask * 255.0
        scorec_tf_fcbox = output_facebox_predection(sess, img_w_mask_255, self.L2_reg_fcbox)
        scorec_tf_fcbox = tf.reshape(scorec_tf_fcbox,[-1,2])
        facebox_loss_total = config.apply_facebox_loss(scorec_tf_fcbox, bb)
#        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#        print(sess.run(tf.gradients(facebox_loss_total,img_w_mask_255)))
#        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        tf.summary.scalar('fcbox_loss_total',facebox_loss_total)
        
        ######################test0922for  dsfd#################################
        for i in range(img_w_mask_255.shape.as_list()[0]):
            single_tf = tf.expand_dims(img_w_mask_255[i], 0)
            part_outscores_dsfd = output_dsfd_predection(sess, single_tf, L2_reg_dsfd)
            part_loss_dsfd = config.apply_dsfd_loss(part_outscores_dsfd, bb)
            tf.summary.scalar('dsfd_loss_'+str(i),part_loss_dsfd)
            dsfd_loss.append(part_loss_dsfd)
        total_loss_dsfd = tf.reduce_sum(dsfd_loss)

        tf.summary.scalar('dsfd_loss_total',total_loss_dsfd)
        
            
        
        with tf.Session() as sess: 
            siximgs_predictions = output_baidu_predection(img_w_mask_255)
            for i in range(len(siximgs_predictions)):
                for j in range(len(siximgs_predictions[i])):
#                    if(config.apply_pnet_loss(siximgs_predictions[i][j], bb)>0.001):
                    part_loss = config.apply_pyradmid_loss(siximgs_predictions[i][j], bb)
#                    part_loss = tf.math.maximum(part_loss - 0.005, 0.0)
                    
                    loss_bd.append(part_loss) 
                    loss_bd_pic.append(part_loss) 
                loss_bd_for_pic_view = tf.reduce_sum(loss_bd_pic)
                tf.summary.scalar('loss_baidu_' + str(i),loss_bd_for_pic_view)  
                loss_bd_pic = []
                
        loss_bd_for_view = tf.reduce_sum(loss_bd)
        tf.summary.scalar('loss_baidu',loss_bd_for_view)  


#        facebox_outpred_tf = output_facebox_predection(img_w_mask_255)
#        for i in range(len(facebox_outpred_tf)):
#            facebox_loss_part = config.apply_pnet_loss(facebox_outpred_tf[i][0], bb)
#            facebox_loss_total.append(facebox_loss_part)
#        facebox_loss_total = tf.reduce_sum(facebox_loss_total)
#        facebox_loss_total = config.apply_pnet_loss(score_tf[0], bb)
        

        # Calculate loss for each patch and do FGSM
        for i, key in enumerate(self.pm.patches.keys()):
            mask_tf = self.pm.patches[key].mask_tf
            
            multiplier = tf.cast((eps <= 55/255.0), tf.float32)
            patch_loss_total = multiplier * config.apply_patch_loss(mask_tf, i, key)
            #total_loss = tf.identity(total_loss_dsfd + pnet_loss_total + loss_bd + facebox_loss_total + patch_loss_total, name="total_loss")
            total_loss = tf.identity(total_loss_dsfd + loss_bd_for_view + facebox_loss_total + pnet_loss_total + patch_loss_total, name="total_loss")
            tf.summary.scalar('Total_Loss',total_loss)  
            grad_raw = tf.gradients(total_loss, mask_tf)[0]
            new_moment = mu * self.accumulators[i] + grad_raw / tf.norm(grad_raw, ord=1)
            assign_op1 = tf.assign(self.accumulators[i], new_moment) 
            moment_assign_op.append(assign_op1)
            new_mask = tf.clip_by_value(mask_tf - eps * tf.sign(self.accumulators[i]), 0.0, 1.0)
            assign_op2 = tf.assign(self.pm.patches[key].mask_tf, new_mask)
            mask_assign_op.append(assign_op2)
            assign_op3 = tf.assign(self.pm.patches[key].grad, grad_raw)#测试是否计算得到梯度
            grad_tf_op.append(assign_op3)
#            with tf.Session() as sess: 
#                mask_tf = self.pm.patches[key].grad
#                print("grad值:",sess.run(mask_tf))              
#            feed_dict = self.lr_schedule(0)
#            sess.run(assign_op1, feed_dict=feed_dict)
#            sess.run(assign_op2, feed_dict=feed_dict)    
#            
#            mask_tf = self.pm.patches[key].mask_tf
#            print("mask_tf值:",sess.run(tf.clip_by_value(mask_tf, 0.0, 1.0)))  
#            mask_assign_op.append(assign_op2)
#            tf.assign()

        # Return assign operation for each patch
        self.mask_assign_op = tuple(mask_assign_op)
        self.moment_assign_op = tuple(moment_assign_op)
        self.grad_tf_op = tuple(grad_tf_op)

    # ===================================================
    # Schedule *learning rate* so that opt process gets better
    # ===================================================
    def lr_schedule(self, i):
        if (i < 100):
            feed_dict = {self.eps: 60/255.0, self.mu: 0.9}
        if (i >= 100 and i < 300):
            feed_dict = {self.eps: 30/255.0, self.mu: 0.9}
        if (i >= 300 and i < 1000):
            feed_dict = {self.eps: 15/255.0, self.mu: 0.95}
        if (i >= 1000):
            feed_dict = {self.eps: 1/255.0,  self.mu: 0.99}
        return feed_dict

    def train(self, sess, i):
        feed_dict = self.lr_schedule(i)
        sess.run(self.moment_assign_op, feed_dict=feed_dict)
        sess.run(self.mask_assign_op, feed_dict=feed_dict)
        sess.run(self.grad_tf_op, feed_dict=feed_dict)
#        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
#        print("梯度值:",sess.run(self.pm.patches[0].grad, feed_dict=feed_dict))        

    # ===================================================
    # Set of aux functions to be used for evaluating and init
    # ===================================================
    def eval(self, sess, dir):
        path_info = "output_img" + dir + "/"
        shutil.rmtree(path_info, ignore_errors=True)
        os.makedirs(path_info)
        self.eval_masks(sess, path_info)
        self.eval_img(sess, path_info)
        self.test_grad(sess, path_info)

    def eval_masks(self, sess, dir):
        global mask_value
        for key in self.pm.patches.keys():
            mask_tf = self.pm.patches[key].mask_tf
            mask = (mask_tf.eval(session=sess) * 255).astype(np.uint8)
            imageio.imsave(dir + key + ".png", mask)
            mask_value = mask

    def eval_img(self, sess, dir):
        width = int(self.pm.imgs_tf.shape[2])
        bs = int(self.pm.imgs_tf.shape[0])
        imgs = (self.img_hat.eval(session=sess) * 255).astype(np.uint8)
        for i in range(bs):
            img = imgs[i]
            imageio.imsave(dir + "attacked" + str(i + 1) + ".png", img)
    def test_grad(self, sess, dir):
        global grad_test
        for key in self.pm.patches.keys():
            grad_test = self.pm.patches[key].grad.eval(session=sess)

# ===================================================
# $$$$$ Define class for loss manipulation $$$$$$$$$$
# ===================================================
class LossManager:
    def __init__(self):
        self.patch_loss = {}
        self.pnet_loss = {}
        self.facebox_loss = {}
        self.pyradmid_loss = {}
        self.dsfd_loss = {}
    # ===================================================
    # Loss function for classification layer output
    # ===================================================

    # (minimize the max value of output prob map)
    def clf_loss_max(self, clf, bb):
        out = tf.reduce_max(tf.math.maximum(clf[...,1] - 0.5, 0.0), axis=(1, 2))
        return tf.reduce_mean(out)

    # (minimize the sum of squares from output prob map)
    def clf_loss_l2(self, clf, bb):
        out = tf.reduce_sum(tf.math.maximum(clf[...,1] - 0.5, 0.0) ** 2, axis=(1, 2))
        return tf.reduce_mean(out)
    
    def clf_loss_l2_v2(self, clf, bb):
        out = tf.reduce_sum(tf.math.maximum(clf[...,1] - 0.25, 0.0) ** 2, axis=(1, 2))
        return tf.reduce_mean(out)

    def clf_loss_l2_v3(self, clf, bb):
        out = tf.reduce_sum(tf.math.maximum(clf[...,1] - 0.25, 0.0) ** 2)
        return tf.reduce_mean(out)
    
    def clf_loss_l2_dsfd(self, clf, bb):
        out = tf.reduce_sum(tf.math.maximum(clf[0] - 0.25, 0.0) ** 2)
        return tf.reduce_mean(out)    
    
    

    # (minimize the sum of the absolute differences for neighboring pixel-values)
    def tv_loss(self, patch):
        loss = tf.image.total_variation(patch)
        return loss

    # (minimize the area with black color)
    def white_loss(self, patch):
        loss = tf.reduce_sum((1 - patch) ** 2)
        return loss

    # ===================================================
    # Input HxWxC
    # ===================================================
    def reg_patch_loss(self, func, name, coefs):
        self.patch_loss[name] = { 'func' : func, 'coef': coefs }

    # ===================================================
    # Input BSxPHxPW
    # ===================================================
    def reg_pnet_loss(self, func, name, coef):
        self.pnet_loss[name] = { 'func' : func, 'coef': coef }
    
    def reg_facebox_loss(self, func, name, coef):
        self.facebox_loss[name] = { 'func' : func, 'coef': coef }
        
    def reg_dsfd_loss(self, func, name, coef):
        self.dsfd_loss[name] = { 'func' : func, 'coef': coef }    
        
    def reg_pyradmid_loss(self, func, name, coef):
        self.pyradmid_loss[name] = { 'func' : func, 'coef': coef }

    # ===================================================
    # Apply losses
    # ===================================================
    def apply_patch_loss(self, patch, patch_i, key):
        patch_loss = []
        for loss in self.patch_loss.keys():
            with tf.variable_scope(loss):
                c = self.patch_loss[loss]['coef'][patch_i]
                patch_loss.append(c * self.patch_loss[loss]['func'](patch))
            tf.summary.scalar(loss + "/" + key, c * patch_loss[-1])
        return tf.add_n(patch_loss)
    
    def apply_pnet_loss(self, clf, bb):
        pnet_loss = []
        for loss in self.pnet_loss.keys():
            with tf.variable_scope(loss):
                c = self.pnet_loss[loss]['coef']
                pnet_loss.append(c * self.pnet_loss[loss]['func'](clf, bb))
            tf.summary.scalar(loss, c * pnet_loss[-1])
        return tf.add_n(pnet_loss)
    
    def apply_facebox_loss(self, clf, bb):
        facebox_loss = []
        for loss in self.facebox_loss.keys():
            with tf.variable_scope(loss):
                c = self.facebox_loss[loss]['coef']
                facebox_loss.append(c * self.facebox_loss[loss]['func'](clf, bb))
            tf.summary.scalar(loss, c * facebox_loss[-1])
        return tf.add_n(facebox_loss)

    def apply_dsfd_loss(self, clf, bb):
        dsfd_loss = []
        for loss in self.dsfd_loss.keys():
            with tf.variable_scope(loss):
                c = self.dsfd_loss[loss]['coef']
                dsfd_loss.append(c * self.dsfd_loss[loss]['func'](clf, bb))
            tf.summary.scalar(loss, c * dsfd_loss[-1])
        return tf.add_n(dsfd_loss)

    def apply_pyradmid_loss(self, clf, bb):
        pyradmid_loss = []
        for loss in self.pyradmid_loss.keys():
            with tf.variable_scope(loss):
                c = self.pyradmid_loss[loss]['coef']
                pyradmid_loss.append(c * self.pyradmid_loss[loss]['func'](clf, bb))
            tf.summary.scalar(loss, c * pyradmid_loss[-1])
        return tf.add_n(pyradmid_loss)

    def colorizer_wb2rgb(self, patch):
        return tf.image.grayscale_to_rgb(patch)

masks = {
    'left_cheek': [np.zeros((150, 180, 1)), (0, 255, 0), (0, 5, 0)],
    'right_cheek': [np.zeros((150, 180, 1)), (255, 0, 0), (5, 0, 0)],
}
images = ['11041.png','11042.png','11043.png','11044.png','11045.png','11046.png','11047.png','11049.png']
config = LossManager()
tf.reset_default_graph()
adv_mask = TrainMask(gpu_id=2)
adv_mask.add_masks(masks)
adv_mask.add_images(images)
sess = tf.Session()

epochs = 2000
config.reg_pnet_loss(config.clf_loss_l2, 'clf_max', 1)
config.reg_pyradmid_loss(config.clf_loss_l2_v2, 'clf_max_v2', 1)
config.reg_facebox_loss(config.clf_loss_l2_v3, 'clf_max_v3', 1)
config.reg_dsfd_loss(config.clf_loss_l2_dsfd, 'clf_max_v4', 1)
config.reg_patch_loss(config.tv_loss, 'tv_loss', [1e-5, 1e-5])

# Do not forget to analyze the sizes that are suitable for
# your resolution
adv_mask.set_input_sizes([(73, 129), (103, 182), (52, 92)])
adv_mask.build(sess)
adv_mask.build_train(sess, config)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('D://TensorBoard//log',sess.graph)

#_init_uninit_vars(sess)
variables_to_restore1 = slim.get_variables_to_restore(include=pyramid_var1)
variables_to_restore2 = slim.get_variables_to_restore(include=pyramid_var2)
variables_to_restore3 = slim.get_variables_to_restore(include=pyramid_var3)
ckpt_filename1 = 'model/epoch_181L2_0.0005.ckpt'
ckpt_filename2 = 'pyramidbox/model/pyramidbox.ckpt'
ckpt_filename3 = 'dsfd/model/epoch_59L2_0.00025.ckpt'
saver1 = tf.train.Saver(variables_to_restore1)
saver1.restore(sess, ckpt_filename1) 

saver2 = tf.train.Saver(variables_to_restore2)
saver2.restore(sess, ckpt_filename2) 

saver3 = tf.train.Saver(variables_to_restore3)
saver3.restore(sess, ckpt_filename3) 

for i in range(epochs):
    # saver = tf.train.Saver(variables_to_restore2)
    # saver.restore(sess, ckpt_filename2) 
    # saver = tf.train.Saver(variables_to_restore3)
    # saver.restore(sess, ckpt_filename3)
    print(str(i + 1) + "/" + str(epochs), end='\r')
    adv_mask.train(sess, i)
    feed_dict = adv_mask.lr_schedule(i)
    summary = sess.run(merged_summary, feed_dict)
    writer.add_summary(summary, i)
    if i%100 == 0:
        adv_mask.eval(sess, str(i + 1))
#    if i%100 == 0:
#        adv_mask.eval(sess, "")
    

writer.flush()
adv_mask.eval(sess, "")
sess.close()