from tensorflow.keras.models import load_model
import numpy as np
import pickle as pkl
from jMetalPy.jmetal.algorithm.multiobjective import NSGAII
from jMetalPy.jmetal.operator import SBXCrossover, PolynomialMutation
from jMetalPy.jmetal.problem.multiobjective.PlausibleCounterfactualProblems \
    import CustomPlausibleCounterfactualProblem
from jMetalPy.jmetal.util.termination_criterion import StoppingByEvaluations
import argparse

parser = argparse.ArgumentParser(description='This code is prepared to run '
                                             'with conditional gan saved '
                                             'with keras.'
                                             'To change the problem '
                                             'definition go to '
                                             './jMetalPy/jmetal/problem/multiobjective/PlausibleCounterfactualProblems.py')
parser.add_argument('-discriminator',
                    help='choose the path for the discriminator path (x.h5)')
parser.add_argument('-generator',
                    help='choose the path for the generator path (x.h5)')
parser.add_argument('-classifier',
                    help='choose the path for the classifier path (x.h5)')
parser.add_argument('-ref_image',
                    help='choose the path for the ref_image (.npy)')
parser.add_argument('-ref_pred',
                    help='choose the path for the ref_pred (.npy)')
parser.add_argument('-attempts', type=int,
                    help='number of attempts')
parser.add_argument('-label_count', type=int,
                    help='number of labels')
parser.add_argument('-condition', type=int,
                    help='chose the conditional input [0-9]')

args = parser.parse_args()

discriminator = load_model(args.discriminator)
generator = load_model(args.generator)
classifier = load_model(args.classifier)
real_image = np.load(args.ref_image)
real_pred = np.load(args.ref_pred)
attempts = args.attempts
label_count = args.label_count
condition = args.condition

for i in range(attempts):
    print(str(i) + '/' + str(attempts) + ' ...')
    noise = np.random.rand(1, 100)
    condition = np.array(condition).reshape(1, 1)

    g = generator([noise, condition])

    d_real = discriminator([real_image.reshape(1, 28, 28, 1), condition])

    problem = CustomPlausibleCounterfactualProblem(image=real_image,
                                                   code=noise,
                                                   decoder=generator,
                                                   discriminator=discriminator,
                                                   classifier=classifier,
                                                   original_pred=real_pred,
                                                   condition=condition,
                                                   label_count=label_count)

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
