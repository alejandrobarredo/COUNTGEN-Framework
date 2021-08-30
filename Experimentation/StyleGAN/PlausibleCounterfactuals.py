import pickle as pkl
import numpy as np
import dnnlib
import os
import dnnlib.tflib as tflib
import pretrained_networks
from jMetalPy.jmetal.algorithm.multiobjective import NSGAII
from jMetalPy.jmetal.operator import SBXCrossover, PolynomialMutation
from jMetalPy.jmetal.problem.multiobjective.PlausibleCounterfactualProblems \
    import StyleGan2PlausibleCounterfactualProblem
from jMetalPy.jmetal.util.termination_criterion import StoppingByEvaluations
from tensorflow.keras.models import load_model

network_pkl = './models/stylegan2-church-config-a.pkl'
seed = 600
truncation_psi = 0.5
print('Loading networks from "%s"...' % network_pkl)
generated_images = []
_G, D, Gs = pretrained_networks.load_networks(network_pkl)
classifier = load_model('./models/classifier_model.h5')
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                  nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False
if truncation_psi is not None:
    Gs_kwargs.truncation_psi = truncation_psi



print('Generating image for seed %d ...' % (seed))
rnd = np.random.RandomState(seed)
code = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]


tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in
                    noise_vars})  # [height, width]
image = Gs.run(code, None, **Gs_kwargs)  # [minibatch, height, width, channel]


pred = np.squeeze(classifier.predict(image))
for i in range(10):
    print(str(i) + '/10 ...')
    problem = StyleGan2PlausibleCounterfactualProblem(image=image,
                                                      code=code,
                                                      decoder=Gs,
                                                      discriminator=D,
                                                      classifier=classifier,
                                                      original_pred=pred,
                                                      Gs_kwargs=Gs_kwargs)

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=20000)
    )

    algorithm.run()
    front = algorithm.get_result()

    with open('/home/xai/Documentos/StyleGAN2/Counterfactuals/Front_' + str(
            i) + '.pkl', 'wb') as f:
        pkl.dump(front, f, pkl.HIGHEST_PROTOCOL)
