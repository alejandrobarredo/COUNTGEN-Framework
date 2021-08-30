import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import os, time, itertools, imageio, pickle
import numpy as np

import pickle as pkl
from jMetalPy.jmetal.algorithm.multiobjective import NSGAII
from jMetalPy.jmetal.operator import SBXCrossover, PolynomialMutation
from jMetalPy.jmetal.problem.multiobjective.PlausibleCounterfactualProblems \
    import MnistDcGanPlausibleCounterfactualProblem
from jMetalPy.jmetal.util.termination_criterion import StoppingByEvaluations

discriminator = load_model('./Models/discriminator.h5')
generator = load_model('./Models/generator.h5')
classifier = load_model('./Models/classifier_model_mnist_unfinished.h5')

for i in range(10):
    print(str(i) + '/10 ...')
    noise = np.random.rand(1, 100)
    condition = np.array(2).reshape(1, 1)

    g = generator([noise, condition])
    real_image = g.numpy().reshape(28, 28)
    d_real = discriminator([real_image.reshape(1, 28, 28, 1), condition])

    pred_orig = classifier.predict(real_image.reshape(1, 28, 28, 1))

    problem = MnistDcGanPlausibleCounterfactualProblem(image=real_image,
                                                       code=noise,
                                                       decoder=generator,
                                                       discriminator=discriminator,
                                                       classifier=classifier,
                                                       original_pred=pred_orig,
                                                       condition=condition)

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables,
            distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=20000)
    )

    algorithm.run()
    front = algorithm.get_result()

    with open(
            './Counterfactuals_unfinished/Front_' + str(
                i) + '.pkl', 'wb') as f:
        pkl.dump(front, f, pkl.HIGHEST_PROTOCOL)
    print('Saved Front_' + str(i))
