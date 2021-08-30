from math import pi, sin, sqrt

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from scipy.spatial import distance
import argparse
from functools import partial
import json
import traceback


import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

import data
import models

"""
.. module:: UF
   :platform: Unix, Windows
   :synopsis: Problems of the CEC2009 multi-objective competition
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class AttGANProblem(FloatProblem):
    """
    Problem of finding counterfactuals for the AttGAN model
    """

    def __init__(self, number_of_variables: int = 30, img=None,
                 code=None,  decoder=None,
                 discriminator=None, classifier=None, sess=None):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

        self.img = img
        self.code = code
        self.decoder = decoder
        self.discriminator = discriminator
        self.classifier = classifier
        self.session = sess

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        sess = self.sess
        code = self.code
        new_code = solution.variables
        img_code_input = self.img
        decoder = self.decoder
        discriminator = self.discriminator
        classifier = self.classifier

        new_img = sess.run(decoder, feed_dict={img_code: img_code_input,
                                               att_code_input: new_code})
        dis_res = sess.run(discriminator, feed_dict={img_input: new_img})
        cla_res = sess.run(classifier, feed_dict={img_input: new_img})

        solution.objectives[0] = distance.braycurtis(code, new_code)
        solution.objectives[1] = dis_res
        solution.objectives[2] = cla_res

        return solution

    def get_name(self):
        return 'AttGAN'