import torch
from vendi_sampling.Model_Systems.model_systems import PrinzEnergy
from vendi_sampling.Model_Systems.samplers import VendiSamp
from vendi_sampling.Model_Systems.helper import logvendi_loss


def test_everything():
    Prinz = PrinzEnergy()

    replicas = 8
    dim = 3
    x_init = torch.rand((replicas, dim), requires_grad=True)
    samples, weights = VendiSamp(E=Prinz.energy, score=logvendi_loss,
                                 steps=10000, x_init=x_init)
    print(samples.shape)
    print(weights.shape)
