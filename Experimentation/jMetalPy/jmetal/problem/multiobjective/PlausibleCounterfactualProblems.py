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


class BicycleGanPlausibleCounterfactualProblem(FloatProblem):
    """
    Problem of finding counterfactuals for the AttGAN model
    """

    def __init__(self, number_of_variables: int = 8, image=None, drawing=None,
                 code=None,  decoder=None,
                 discriminator_0=None, discriminator_1=None,
                 classifier=None, original_pred=None):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [-10.0]
        self.upper_bound = self.number_of_variables * [10.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

        self.image = image
        self.drawing = drawing
        self.code = code
        self.decoder = decoder
        self.original_pred = original_pred
        original_pred_decision = 0
        if original_pred >= 0.5:
            original_pred_decision = 1
        self.original_pred_decision = original_pred_decision
        self.discriminator_0 = discriminator_0
        self.discriminator_1 = discriminator_1
        self.classifier = classifier
        self.counterfactuals = pd.DataFrame(data=[], columns=[
            'CounterfactualImage', 'ClassifierProbability',
            'DiscriminatorDistance', 'OriginalCode', 'CounterfactualCode'])

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        df = self.counterfactuals
        real_input_image = self.image
        real_input_drawing = self.drawing
        code = self.code
        new_code = np.array(solution.variables).reshape(1, 8).astype(
            np.float32)

        decoder = self.decoder
        discriminator_0 = self.discriminator_0
        discriminator_1 = self.discriminator_1
        classifier = self.classifier

        output = decoder(torch.from_numpy(real_input_drawing.reshape(1, 1,
                                                                     256,
                                                                     256)),
                         torch.from_numpy(new_code))

        disc_o_real = discriminator_0(torch.from_numpy(
            real_input_image.reshape(1, 3, 256, 256)))
        disc_o_real = discriminator_1(disc_o_real, True)

        disc_o_fake = discriminator_0(output)
        disc_o_fake = discriminator_1(disc_o_fake, True)

        # Extract values
        output = output.cpu().detach().numpy()
        disc_o_real = disc_o_real[0].cpu().detach().numpy()
        disc_o_fake = disc_o_fake[0].cpu().detach().numpy()

        disc_decision = disc_o_fake > disc_o_real

        # Prep output fopr classifier
        output = output[0, :, :, :]
        output = np.rollaxis(output, 0, 3)

        pred = classifier.predict(output.reshape(1, 256, 256, 3))
        solution.prediction = pred
        pred_decision = 0
        if pred >= 0.5:
            pred_decision = 1

        code_difference = np.sum(np.power(new_code - code, 2))
        pred_difference = self.original_pred - pred
        disc_difference = disc_o_real - disc_o_fake

        # print('Code difference: ' + str(code_difference))
        # print('Old class: ' + str(self.original_pred))
        # print('New class: ' + str(int(pred)) + '(' + str(np.squeeze(pred)) +
        #                                                  ')')
        # print('Gan decision: ' + str(disc_decision))
        # plt.imshow(output)
        # plt.pause(0.1)

        solution.objectives[0] = code_difference
        solution.objectives[1] = 1 - np.squeeze(pred_difference)
        solution.objectives[2] = 1 - disc_difference

        ploted_img = (output - np.min(output)) / (np.max(output) -
                                                  np.min(output))
        solution.counterfactualImage = ploted_img

        if pred_decision != self.original_pred_decision and (disc_o_fake <
                                                     disc_o_real):
            df.loc[df.shape[0]] = [output, pred, disc_difference, code, new_code]
            # print(df.shape)
            # print('Counterfactual')
            # print('Code difference: ' + str(code_difference))
            # print('Old class: ' + str(int(np.squeeze(self.original_pred))))
            # print('New class: ' + str(int(pred)) + '(' + str(np.squeeze(pred)) +
            #                                                  ')')
            # print('Discriminator fake/real: ' + str(disc_o_fake) + '/' + str(
            #     disc_o_real))

            plt.imshow(ploted_img)
            # plt.pause(0.1)
            files = os.listdir('/home/xai/Documentos/BicycleGAN/datasets'
                               '/shoes_dataset/Counterfactuals/Images/')
            plt.savefig('/home/xai/Documentos/BicycleGAN/datasets'
                        '/shoes_dataset/Counterfactuals/Images/Count' +
                        str(len(files)) + '.png')
            df.to_pickle('/home/xai/Documentos/BicycleGAN/datasets'
                         '/shoes_dataset/Counterfactuals'
                         '/counterfactuals.pkl')
        return solution

    def get_name(self):
        return 'PlausibleCounterfactual'


class StyleGan2PlausibleCounterfactualProblem(FloatProblem):
    """
    Problem of finding counterfactuals for the AttGAN model
    """

    def __init__(self, number_of_variables: int = 512, image=None,
                 code=None,  decoder=None,
                 discriminator=None,
                 classifier=None,
                 original_pred=None,
                 Gs_kwargs=None):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [-10.0]
        self.upper_bound = self.number_of_variables * [10.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

        self.image = image
        self.code = code
        self.decoder = decoder
        self.original_pred = original_pred
        original_pred_decision = 0
        if original_pred >= 0.5:
            original_pred_decision = 1
        self.original_pred_decision = original_pred_decision
        self.discriminator = discriminator
        self.classifier = classifier
        self.Gs_kwargs = Gs_kwargs
        self.counterfactuals = pd.DataFrame(data=[], columns=[
            'CounterfactualImage', 'ClassifierProbability',
            'DiscriminatorDistance', 'OriginalCode', 'CounterfactualCode'])

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        df = self.counterfactuals
        real_input_image = self.image
        code = self.code
        Gs_kwargs = self.Gs_kwargs
        new_code = np.array(solution.variables).reshape(1, self.number_of_variables).astype(
            np.float32)

        decoder = self.decoder
        discriminator = self.discriminator
        classifier = self.classifier

        new_image = decoder.run(new_code, None, **Gs_kwargs)
        images = np.concatenate([real_input_image, new_image], axis=0)
        new_pred = classifier.predict(new_image)
        solution.prediction = new_pred

        d_result = discriminator.run(np.rollaxis(images, 3, 1), None)

        pred_decision = 0
        if new_pred >= 0.5:
            pred_decision = 1

        code_difference = np.sum(np.power(new_code - code, 2))
        pred_difference = self.original_pred - new_pred
        disc_difference = d_result[0] - d_result[1]

        solution.objectives[0] = code_difference
        solution.objectives[1] = 1 - np.squeeze(pred_difference)
        solution.objectives[2] = 1 - disc_difference

        solution.counterfactualImage = new_image[0, :, :, :]

        if (pred_decision != self.original_pred_decision) and (disc_difference
                                                             < 0):
            # plt.imshow(new_image[0, :, :, :])
            plt.pause(0.1)
            files = os.listdir('/home/xai/Documentos/StyleGAN2/Counterfactuals/')
            plt.savefig('/home/xai/Documentos/StyleGAN2/Counterfactuals/Count' +
                        str(len(files)) + '.png')
        return solution


    def get_name(self):
        return 'PlausibleCounterfactual'


class ThreeDGanPlausibleCounterfactualProblem(FloatProblem):
    """
    Problem of finding counterfactuals for the AttGAN model
    """

    def __init__(self, number_of_variables: int = 200, image=None,
                 code=None,  decoder=None,
                 discriminator=None,
                 classifier=None,
                 original_pred=None,
                 session=None,
                 z=None,
                 real_models=None):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [-10.0]
        self.upper_bound = self.number_of_variables * [10.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

        self.image = image
        self.code = code
        self.decoder = decoder
        self.original_pred = original_pred
        original_pred_decision = 0
        if original_pred >= 0.5:
            original_pred_decision = 1
        self.original_pred_decision = original_pred_decision
        self.discriminator = discriminator
        self.classifier = classifier
        self.z = z
        self.real_models = real_models
        self.session = session
        self.counterfactuals = pd.DataFrame(data=[], columns=[
            'CounterfactualImage', 'ClassifierProbability',
            'DiscriminatorDistance', 'OriginalCode', 'CounterfactualCode'])

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        df = self.counterfactuals
        real_image = self.image
        code = self.code
        session = self.session
        new_code = np.array(solution.variables).reshape(1, self.number_of_variables).astype(
            np.float32)
        z = self.z
        real_models = self.real_models
        G_Fake = self.decoder
        D_Real, D_Fake = self.discriminator
        classifier = self.classifier

        new_image = session.run(G_Fake, feed_dict={z: code})
        new_pred = classifier.predict(new_image)
        solution.prediction = new_pred

        d_real = session.run(D_Real, feed_dict={real_models: real_image})[0][0]
        d_fake = session.run(D_Fake, feed_dict={G_Fake: new_image})[0][0]

        pred_decision = 0
        if new_pred >= 0.5:
            pred_decision = 1

        code_difference = np.sum(np.power(new_code - code, 2))
        pred_difference = self.original_pred - new_pred
        disc_difference = d_real - d_fake

        solution.objectives[0] = code_difference
        solution.objectives[1] = 1 - np.squeeze(pred_difference)
        solution.objectives[2] = 1 - disc_difference

        solution.counterfactualImage = new_image[0, :, :, :]

        if (pred_decision != self.original_pred_decision) and (disc_difference
                                                               < 0):
            # plt.imshow(new_image[0, :, :, :])
            self.plot_3d_image(real_image)
            self.plot_3d_image(new_image)
            files = os.listdir('/home/xai/Documentos/StyleGAN2/Counterfactuals/')
            plt.savefig('/home/xai/Documentos/StyleGAN2/Counterfactuals/Count' +
                        str(len(files)) + '.png')
        return solution


    def get_name(self):
        return 'PlausibleCounterfactual'

    @staticmethod
    def plot_3d_image(voxel_data, threshold=0.6, show=0):
        voxel_data[voxel_data > threshold] = 1
        voxel_data[voxel_data <= threshold] = 0
        bool_image = voxel_data.astype(bool)

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(bool_image)
        if show == 1:
            plt.show()
        else:
            plt.pause(0.1)


class AttGanPlausibleCounterfactualProblem(FloatProblem):
    """
    Problem of finding counterfactuals for the AttGAN model
    """

    def __init__(self, number_of_variables: int = 13, image=None,
                 code=None,  decoder=None,
                 discriminator=None,
                 classifier=None,
                 original_pred=None,
                 session=None,
                 placeholders=None):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]

        self.image = image
        self.code = code
        self.decoder = decoder
        self.original_pred = original_pred
        original_pred_decision = 0
        if original_pred >= 0.5:
            original_pred_decision = 1
        self.original_pred_decision = original_pred_decision
        self.discriminator = discriminator
        self.classifier = classifier
        self.session = session
        self.placeholders = placeholders
        self.counterfactuals = pd.DataFrame(data=[], columns=[
            'CounterfactualImage', 'ClassifierProbability',
            'DiscriminatorDistance', 'OriginalCode', 'CounterfactualCode'])

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        df = self.counterfactuals
        real_image = self.image
        code = self.code
        sess = self.session
        img_input, att_code_input = self.placeholders
        new_code = np.array(solution.variables).reshape(1, self.number_of_variables).astype(
            np.float32)

        img_retrieved = self.decoder
        real_fake = self.discriminator
        classifier = self.classifier

        new_image = sess.run(img_retrieved,
                             feed_dict={img_input: real_image,
                                        att_code_input: new_code})
        new_image_n = self.normalize(new_image)
        new_pred = classifier.predict(new_image_n)[0][0]
        solution.prediction = new_pred

        d_real, _ = sess.run(real_fake, feed_dict={img_retrieved:
                                                       real_image})
        d_real = d_real[0][0]
        d_fake, _ = sess.run(real_fake, feed_dict={img_retrieved: new_image})
        d_fake = d_fake[0][0]

        pred_decision = 0
        if new_pred >= 0.5:
            pred_decision = 1

        code_difference = np.sum(np.power(new_code - code, 2))
        pred_difference = self.original_pred - new_pred
        disc_difference = d_real - d_fake

        solution.objectives[0] = code_difference.astype(np.float32)
        solution.objectives[1] = (1 - np.squeeze(pred_difference)).astype(np.float32)
        solution.objectives[2] = (1 - disc_difference).astype(np.float32)

        solution.counterfactualImage = new_image[0, :, :, :]

        if (pred_decision != self.original_pred_decision) and (disc_difference
                                                               < 0):
            f, ax = plt.subplots(1, 2)
            ax[0].imshow(self.normalize(real_image[0, :, :, :]))
            ax[1].imshow(self.normalize(new_image[0, :, :, :]))
            files = os.listdir('./Counterfactuals/images/')
            plt.savefig('./Counterfactuals/images/Count' +
                        str(len(files)) + '.png')
            plt.close(f)
        return solution


    def get_name(self):
        return 'PlausibleCounterfactual'


    @staticmethod
    def normalize(_img):
        return (_img- np.min(_img))/(np.max(_img) - np.min(_img))


class MnistDcGanPlausibleCounterfactualProblem(FloatProblem):
    """
    Problem of finding counterfactuals for the AttGAN model
    """

    def __init__(self, number_of_variables: int = 100, image=None,
                 code=None,  decoder=None,
                 discriminator=None,
                 classifier=None,
                 original_pred=None,
                 condition=None):
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


    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        real_image = self.image
        real_image = real_image.reshape(1, 28, 28, 1)
        code = self.code
        condition = self.condition
        new_code = np.array(solution.variables).reshape(1, self.number_of_variables).astype(
            np.float32)
        real_labels = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        new_condition = np.random.choice(real_labels, 1)

        generator = self.decoder
        discriminator = self.discriminator
        classifier = self.classifier

        new_image = generator([new_code, new_condition])
        new_image = new_image.numpy().reshape(1, 28, 28, 1)
        new_pred = np.squeeze(classifier.predict(new_image))
        # heat_map = grad_cam_plus(classifier, new_image.reshape(28, 28, 1),
        #                          layer_name=classifier.layers[0].name,
        #                          label_name=['0', '1', '2', '3', '4', '5', '6',
        #                                      '7', '8', '9'])
        solution.prediction = new_pred
        #solution.heat_map = heat_map

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

        if (new_pred != original_pred) and (disc_difference < 0):
            f, ax = plt.subplots(1, 2)
            ax[0].imshow(self.normalize(real_image[0, :, :, 0]))
            ax[1].imshow(self.normalize(new_image[0, :, :, 0]))
            # plt.show()
            files = os.listdir('./Counterfactuals/images/')
            plt.savefig('./Counterfactuals/images/Count' +
                        str(len(files)) + '.png')
            plt.close(f)
        return solution


    def get_name(self):
        return 'PlausibleCounterfactual'


    @staticmethod
    def normalize(_img):
        return (_img- np.min(_img))/(np.max(_img) - np.min(_img))