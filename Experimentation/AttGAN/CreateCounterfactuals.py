from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import sys

import numpy as np
import pickle as pkl
from jMetalPy.jmetal.algorithm.multiobjective import NSGAII
from jMetalPy.jmetal.operator import SBXCrossover, PolynomialMutation
from jMetalPy.jmetal.problem.multiobjective.PlausibleCounterfactualProblems \
    import AttGanPlausibleCounterfactualProblem

from jMetalPy.jmetal.util.termination_criterion import StoppingByEvaluations

from functools import partial
import json
import traceback

import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl
import os

import data
import models
import matplotlib.pyplot as plt
from keras.models import load_model


def normalize(_img):
    return (_img- np.min(_img))/(np.max(_img) - np.min(_img))


# =============================================================================
# =                                    param                                  =
# =============================================================================
# model
atts = ["Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
        "Bushy_Eyebrows", "Eyeglasses", "Male", "Mouth_Slightly_Open",
        "Mustache", "No_Beard", "Pale_Skin", "Young"]

n_att = len(atts)
img_size = 128
shortcut_layers = 1
inject_layers = 1
enc_dim = 64
dec_dim = 64
dis_dim = 64
dis_fc_dim = 1024
enc_layers = 5
dec_layers = 5
dis_layers = 5
# testing
thres_int = 0.5
test_int = 1.0
# others
use_cropped_img = False
experiment_name = "128_shortcut1_inject1_none"
classifier_name = 'Celeba-Gender-Classifier'


# =============================================================================
# =                                  GRAPH                                    =
# =============================================================================

# Sesiones
sess = tl.session()
# sess_1 = tl.session()

# data
test_data = data.Celeba('/home/xai/Descargas/celeba/', atts,
                        img_size, 1,
                        part='test',
                        sess=sess,
                        crop=True)

# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers)

Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers,
               shortcut_layers=shortcut_layers, inject_layers=inject_layers)

D = partial(models.D,  n_att=n_att, dim=dis_dim,
            fc_dim=dis_fc_dim, n_layers=dis_layers)

# inputs
img_input = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
att_code_input = tf.placeholder(tf.float32, shape=[None, n_att])

# sample
img_code = Genc(img_input, is_training=False)
img_retrieved = Gdec(Genc(img_input), att_code_input, is_training=False)
real_fake = D(img_retrieved)

# =============================================================================
# =                                 LOAD                                      =
# =============================================================================

dirs = ['./output/%s/checkpoints' % experiment_name,
        './output/%s/checkpoints' % classifier_name]

for dir in dirs:
    vars_to_load = []

    for i in tf.train.list_variables(dir):
        vars_to_load.append(i[0])

    assignment_map = {variable.op.name: variable for variable in
                      tf.global_variables() if variable.op.name in vars_to_load}
    tf.train.init_from_checkpoint(dir, assignment_map)
sess.run(tf.global_variables_initializer())

classifier = load_model('./models/classifier_model.h5')

# =============================================================================
# =                                    TEST                                   =
# =============================================================================

# sample
try:
    for i in range(10):
        print(str(i) + '/10 ...')
        batch = test_data.next()
        # We load a sample from the test set
        _img_input = batch[0]

        _att_code_input = batch[1]

        # We check if the retrieved image fools the classifier

        _img_code = sess.run(img_code, feed_dict={img_input: _img_input})

        _img_retrieved = sess.run(img_retrieved,
                                  feed_dict={img_input: _img_input,
                                             att_code_input: _att_code_input})
        plt.imshow(_img_retrieved.reshape(128, 128, 3))
        plt.show()
        #
        # _real_fake_orig = sess.run(real_fake,
        #                            feed_dict={img_retrieved: _img_input})
        # _real_fake, _ = sess.run(real_fake,
        #                          feed_dict={img_retrieved: _img_retrieved})
        #
        # _img_input_n = normalize(_img_input)
        # _img_retrieved_n = normalize(_img_retrieved)
        # _prediction_orig = classifier.predict(_img_input_n)[0][0]
        # _prediction = classifier.predict(_img_retrieved_n)[0][0]
        #
        #
        # problem = AttGanPlausibleCounterfactualProblem(image=_img_input,
        #                                                code=_att_code_input,
        #                                                decoder=img_retrieved,
        #                                                discriminator=real_fake,
        #                                                classifier=classifier,
        #                                                original_pred=_prediction_orig,
        #                                                session=sess,
        #                                                placeholders=(
        #                                                    img_input,
        #                                                    att_code_input))
        #
        # algorithm = NSGAII(
        #     problem=problem,
        #     population_size=100,
        #     offspring_population_size=100,
        #     mutation=PolynomialMutation(
        #         probability=1.0 / problem.number_of_variables,
        #         distribution_index=20),
        #     crossover=SBXCrossover(probability=1.0, distribution_index=20),
        #     termination_criterion=StoppingByEvaluations(max_evaluations=20000)
        # )
        #
        # algorithm.run()
        # front = algorithm.get_result()
        #
        # with open(
        #         './Counterfactuals/Front_' + str(
        #                 i) + '.pkl', 'wb') as f:
        #     pkl.dump(front, f, pkl.HIGHEST_PROTOCOL)

except:
    traceback.print_exc()
finally:
    sess.close()
