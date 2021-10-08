from jMetalPy.jmetal.operator import PolynomialMutation, BitFlipMutation
from jMetalPy.jmetal.util.comparator import DominanceComparator
from jMetalPy.jmetal.util.evaluator import SequentialEvaluator
from jMetalPy.jmetal.util.generator import RandomGenerator
from jMetalPy.jmetal.util.observable import DefaultObservable
from jMetalPy.jmetal.util.termination_criterion import StoppingByEvaluations


class _Store:

    @property
    def default_observable(self):
        return DefaultObservable()

    @property
    def default_evaluator(self):
        return SequentialEvaluator()

    @property
    def default_generator(self):
        return RandomGenerator()

    @property
    def default_termination_criteria(self):
        return StoppingByEvaluations(max_evaluations=25000)

    @property
    def default_comparator(self):
        return DominanceComparator()

    @property
    def default_mutation(self):
        return {
            'real': PolynomialMutation(probability=0.15, distribution_index=20),
            'binary': BitFlipMutation(0.15)
        }


store = _Store()
