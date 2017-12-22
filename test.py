from estimators import EnsembleNetwork, GaussianLossEstimator, GaussianLearningRateEstimator

#estimators
net = EnsembleNetwork()
gauss = GaussianLossEstimator()
gaus_LR = GaussianLearningRateEstimator()

from ensembles import VanillaEnsemble, BootstrapEnsemble
#ensemples
vaniallaEnsemble = VanillaEnsemble()
BootstrapEnsemble = BootstrapEnsemble()
