import os

from scipy.spatial import distance
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from jMetalPy.jmetal.core.problem import FloatProblem
from jMetalPy.jmetal.core.solution import FloatSolution
from gradCam.gradcam import grad_cam, grad_cam_plus

"""
.. module:: UF
   :platform: Unix, Windows
   :synopsis: Problems of the CEC2009 multi-objective competition
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class CustomPlausibleCounterfactualProblem(FloatProblem):
    """
    Problem of finding counterfactuals for the AttGAN model
    """

    def __init__(self, number_of_variables: int = 100, image=None,
                 code=None,  decoder=None,
                 discriminator=None,
                 classifier=None,
                 original_pred=None,
                 condition=None,
                 label_count=None):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [-5.0]
        self.upper_bound = self.number_of_variables * [5.0]

        self.image = image
        self.code = code
        self.decoder = decoder
        self.original_pred = original_pred[0]
        self.original_pred_decision = np.argmax(original_pred)
        self.discriminator = discriminator
        self.classifier = classifier
        self.condition = condition
        self.label_count = label_count

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        real_image = self.image
        real_image = real_image.reshape(1, 28, 28, 1)
        code = self.code
        condition = self.condition
        new_code = np.array(solution.variables).reshape(1, self.number_of_variables).astype(
            np.float32)
        real_labels = list(range(self.label_count))
        new_condition = np.random.choice(real_labels, 1)

        generator = self.decoder
        discriminator = self.discriminator
        classifier = self.classifier

        new_image = generator([new_code, new_condition])
        new_image = new_image.numpy().reshape(1, 28, 28, 1)
        new_pred = np.squeeze(classifier.predict(new_image))
        heat_map = grad_cam_plus(classifier, new_image.reshape(28, 28, 1),
                                 layer_name=classifier.layers[0].name)
        solution.prediction = new_pred
        solution.heat_map = heat_map

        d_real = discriminator([real_image, condition]).numpy()[0][0]
        d_fake = discriminator([new_image, condition]).numpy()[0][0]

        original_pred = self.original_pred[np.argmax(self.original_pred)]
        new_pred = new_pred[np.argmax(new_pred)]

        code_difference = np.sum(np.power(new_code - code, 2))
        pred_difference = original_pred - new_pred
        disc_difference = d_real - d_fake

        solution.objectives[0] = code_difference.astype(np.float32)
        solution.objectives[1] = (1 - np.squeeze(pred_difference)).astype(np.float32)
        solution.objectives[2] = (1 - disc_difference).astype(np.float32)

        solution.counterfactualImage = new_image[0, :, :, :]

        return solution

    @staticmethod
    def get_name(self):
        return 'PlausibleCounterfactual'

    @staticmethod
    def normalize(_img):
        return (_img - np.min(_img))/(np.max(_img) - np.min(_img))