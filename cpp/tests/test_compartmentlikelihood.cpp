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

#include "memilio/compartments/likelihood.h"
#include "ode_seir/model.h"
#include "gtest/gtest.h"

TEST(TestCompartmentLikelihood, compute)
{
    Eigen::MatrixXd expected_drift(4, 4);
    expected_drift << -2.70416441e-06, 0.00000000e00, -1.08166577e-04, 0.00000000e00, 2.70416441e-06, -1.92307692e-01,
        1.08166577e-04, 0.00000000e00, 0.00000000e00, 1.92307692e-01, -1.66666667e-01, 0.00000000e00, 0.00000000e00,
        0.00000000e00, 1.66666667e-01, 0.00000000e00;

    mio::oseir::Model model;
    model.parameters.set<mio::oseir::TimeExposed>(5.2);
    model.parameters.set<mio::oseir::TimeInfected>(6);
    model.parameters.set<mio::oseir::TransmissionProbabilityOnContact>(1);
    model.parameters.get<mio::oseir::ContactPatterns>().get_baseline()(0, 0) = 1;

    double total_population                                                                            = 8600;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Exposed)}]   = 300;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Infected)}]  = 200;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Recovered)}] = 100;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Susceptible)}] =
        total_population -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Exposed)}] -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Infected)}] -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Recovered)}];


    mio::Likelihood<decltype(model)> likelihood(model);

    // TODO FJ
    double lik = likelihood.compute(ts);
}
