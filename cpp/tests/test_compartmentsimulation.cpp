/* 
* Copyright (C) 2020-2023 German Aerospace Center (DLR-SC)
*
* Authors: Jan Kleinert, Daniel Abele
*
* Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "memilio/compartments/simulation.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(TestCompartmentSimulation, integrator_uses_model_reference)
{
    struct MockModel {
        Eigen::VectorXd get_initial_values() const
        {
            return Eigen::VectorXd::Zero(1);
        }
        void eval_right_hand_side(const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&,
                                  double, Eigen::Ref<Eigen::VectorXd> dydt) const
        {
            dydt[0] = this->m_dydt;
        }
        double m_dydt = 1.0;
    };

    auto sim = mio::Simulation<MockModel>(MockModel(), 0.0);
    sim.advance(1.0);

    ASSERT_NEAR(sim.get_result().get_last_value()[0], 1.0, 1e-5);

    //modifying the model from the outside should affect the integration result
    sim.get_model().m_dydt = 2.0;
    sim.advance(2.0);

    ASSERT_NEAR(sim.get_result().get_last_value()[0], 3.0, 1e-5);
}

TEST(TestCompartmentLikelihood, compute_noise_correlation_at_t0)
{
    Eigen::MatrixXd expected_noise_correlation(4, 4);
    expected_noise_correlation << 2.51550178e-06, -2.51550178e-06, 0.00000000e00, 0.00000000e00, -2.51550178e-06,
        6.71092337e-03, -6.70840787e-03, 0.00000000e00, 0.00000000e00, -6.70840787e-03, 1.05843769e-02, -3.87596899e-03,
        0.00000000e00, 0.00000000e00, -3.87596899e-03, 3.87596899e-03;

    struct MockSeirModel {
        Eigen::VectorXd get_initial_values() const
        {
            Eigen::VectorXd initial_state(4);
            initial_state << 8000, 300, 200, 100;
            return Eigen::VectorXd(4);
        }
        void eval_right_hand_side(const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>& y,
                                  double, Eigen::Ref<Eigen::VectorXd> dydt) const
        {
            double time_exposed             = 5.2;
            double time_infected            = 6.0;
            double transmission_probability = 1.;
            double total_pop                = this->get_initial_values().sum();

            dydt[0] = -transmission_probability / total_pop * y[0] * y[2];
            dydt[1] = -transmission_probability / total_pop * y[0] * y[2] - 1. / time_exposed * y[1];
            dydt[2] = 1. / time_exposed * y[1] - 1. / time_infected * y[2];
            dydt[3] = 1. / time_infected * y[2];
        }
    } model;

    double time_point = 0.;
    auto sim          = mio::Simulation<decltype(model)>(model, time_point);

    Eigen::MatrixXd noise_correlation = mio::compute_noise_correlation(time_point, model, sim);
}
