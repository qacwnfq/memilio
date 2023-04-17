#############################################################################
# Copyright (C) 2020-2021 German Aerospace Center (DLR-SC)
#
# Authors:
#
# Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#############################################################################
import unittest

import numpy as np

from memilio.simulation import Damping
from memilio.simulation.oseir import Index_InfectionState
from memilio.simulation.oseir import InfectionState as State
from memilio.simulation.oseir import Model, simulate, likelihood

from memilio.simulation import TimeSeries


class Test_oseir_integration(unittest.TestCase):

    def setUp(self):

        model = Model()

        model.parameters.TimeExposed.value = 5.2
        model.parameters.TimeInfected.value = 6.
        model.parameters.TransmissionProbabilityOnContact.value = 1.

        model.populations[Index_InfectionState(State.Susceptible)] = 4800
        model.populations[Index_InfectionState(State.Exposed)] = 100
        model.populations[Index_InfectionState(State.Infected)] = 50
        model.populations[Index_InfectionState(State.Recovered)] = 50

        model.parameters.ContactPatterns.baseline = np.ones((1, 1))
        model.parameters.ContactPatterns.minimum = np.zeros((1, 1))
        model.parameters.ContactPatterns.add_damping(
            Damping(coeffs=np.r_[0.9], t=30.0, level=0, type=0))

        model.check_constraints()

        self.model = model

    def test_simulate_simple(self):
        result = simulate(t0=0., tmax=100., dt=0.1, model=self.model)
        self.assertAlmostEqual(result.get_time(0), 0.)
        self.assertAlmostEqual(result.get_time(1), 0.1)
        self.assertAlmostEqual(result.get_last_time(), 100.)

    def test_check_constraints_parameters(self):

        model = Model()

        model.parameters.TimeExposed.value = 5.2
        model.parameters.TimeInfected.value = 6.
        model.parameters.TransmissionProbabilityOnContact.value = 1.

        model.parameters.TimeExposed.value = 5.2
        model.parameters.TimeInfected.value = 6.
        model.parameters.TransmissionProbabilityOnContact.value = 1.
        self.assertEqual(model.parameters.check_constraints(), 0)

        model.parameters.TimeExposed.value = -1.
        self.assertEqual(model.parameters.check_constraints(), 1)

        model.parameters.TimeExposed.value = 5.2
        model.parameters.TimeInfected.value = 0
        self.assertEqual(model.parameters.check_constraints(), 1)

        model.parameters.TimeInfected.value = 6.
        model.parameters.TransmissionProbabilityOnContact.value = -1.
        self.assertEqual(model.parameters.check_constraints(), 1)

    def test_likelihood(self):
        expected_likelihood = 484.27044719538532


        model = Model()

        model.parameters.TimeExposed.value = 5.2
        model.parameters.TimeInfected.value = 6.
        model.parameters.TransmissionProbabilityOnContact.value = 0.25
        model.parameters.ContactPatterns.baseline = np.array([[1]])

        model.populations[Index_InfectionState(State.Susceptible)] = int(8e8)
        model.populations[Index_InfectionState(State.Exposed)] = int(1e5)
        model.populations[Index_InfectionState(State.Infected)] = int(1e5)
        model.populations[Index_InfectionState(State.Recovered)] = int(1e3)

        observations = TimeSeries(4)
        observations.add_time_point(0, np.array([800000000, 100000, 100000, 1000]))
        observations.add_time_point(1, np.array([799974430, 105436, 103089, 18045]))
        observations.add_time_point(2, np.array([799947800, 110548, 106801, 35851]))
        observations.add_time_point(3, np.array([799921183, 115925, 110623, 53269]))
        observations.add_time_point(4, np.array([799891616, 121706, 114866, 72812]))
        observations.add_time_point(5, np.array([799862092, 126753, 119617, 92538]))
        observations.add_time_point(6, np.array([799832577, 132015, 123923, 112485]))
        observations.add_time_point(7, np.array([799799748, 137970, 128836, 134446]))
        observations.add_time_point(8, np.array([799767297, 143827, 133673, 156203]))
        observations.add_time_point(9, np.array([799733809, 149548, 138980, 178663]))
        observations.add_time_point(10, np.array([799700308, 155326, 144198, 201168]))

        actual_likelihood = likelihood(model=model, observations=observations)
        self.assertAlmostEqual(actual_likelihood, expected_likelihood)


if __name__ == '__main__':
    unittest.main()
