import os
from tensorflow.keras.models import load_model
from options.test_options import TestOptions
##from data import create_dataset
#from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from jMetalPy.jmetal.algorithm.multiobjective import NSGAII
from jMetalPy.jmetal.operator import SBXCrossover, PolynomialMutation
from jMetalPy.jmetal.problem.multiobjective.PlausibleCounterfactualProblems \
    import PlausibleCounterfactualProblem
from jMetalPy.jmetal.util.termination_criterion import StoppingByEvaluations



# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle
opt.isTest = 1

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

for i, data in enumerate(islice(dataset, 1)):  # opt.num_test

    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    for nn in [0]:  # range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode

        real_input_drawing = model.real_A
        real_input_image = model.real_B


        code, _ = model.netE(real_input_image)  # Encoder
        output = model.netG(real_input_drawing, code)  # Generator

        decision_Real_x = model.netD2(real_input_image)  # Discriminator
        decision_Real = model.criterionGAN(decision_Real_x, True)
        decision_Fake_x = model.netD2(output)  # Discriminator
        decision_Fake = model.criterionGAN(decision_Fake_x, True)

        real_input_drawing = real_input_drawing.cpu().detach().numpy()
        real_input_image = real_input_image.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        decision_Real_x = decision_Real_x[0].cpu().detach().numpy()
        decision_Fake_x = decision_Fake_x[0].cpu().detach().numpy()
        decision_Real = decision_Real[0].cpu().detach().numpy()
        decision_Fake = decision_Fake[0].cpu().detach().numpy()
        code = code.cpu().detach().numpy()

        real_input_drawing = real_input_drawing[0, :, :, :]
        real_input_drawing = np.rollaxis(real_input_drawing, 0, 3)
        real_input_image = real_input_image[0, :, :, :]
        real_input_image = np.rollaxis(real_input_image, 0, 3)
        with open('real_input_image.pkl', 'wb') as f:
            pkl.dump(real_input_image, f, pkl.HIGHEST_PROTOCOL)

        output = output[0, :, :, :]
        output = np.rollaxis(output, 0, 3)

        if decision_Fake > decision_Real:
            print('Image is real')
        else:
            print('Image is fake')

        classifier = load_model('./datasets/shoes_dataset/Models/model.h5')

        pred = classifier.predict(output.reshape(1, 256, 256, 3))

        if pred < 0.5:
            print('Class is: 0 (' + str(pred) + ')')
        else:
            print('Class is: 1 (' + str(pred) + ')')

        # fig, ax = plt.subplots(1, 4)
        # ax[0].imshow(real_input_drawing)
        # ax[1].imshow(real_input_image)
        # ax[2].imshow(output)

        print(str(i) + '/10 ...')
        problem = PlausibleCounterfactualProblem(image=real_input_image,
                                                 drawing=real_input_drawing,
                                                 code=code,
                                                 decoder=model.netG,
                                                 discriminator_0=model.netD2,
                                                 discriminator_1=model.criterionGAN,
                                                 classifier=classifier,
                                                 original_pred=pred)

        algorithm = NSGAII(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables,
                distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=2000)
        )

        algorithm.run()
        front = algorithm.get_result()

        with open('./Counterfactuals/Front_' + str(i) + '.pkl', 'wb') as f:
            pkl.dump(front, f, pkl.HIGHEST_PROTOCOL)

        if i == 10:
            quit()
