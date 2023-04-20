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

TEST(TestCompartmentLikelihood, computeLikelihoodOseir)
{
    double expectedLikelihood = 38.584080700213570;

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
    // random observations were simulated using stochastic differential equations and pyross
    mio::TimeSeries<double> observations(4);
    Eigen::VectorXd obs0(4);
    obs0 << 8000, 300, 200, 100;
    Eigen::VectorXd obs1(4);
    obs1 << 7796, 438, 221, 145;
    Eigen::VectorXd obs2(4);
    obs2 << 7574, 560, 283, 183;
    Eigen::VectorXd obs3(4);
    obs3 << 7300, 707, 366, 227;
    Eigen::VectorXd obs4(4);
    obs4 << 6977, 888, 450, 285;
    Eigen::VectorXd obs5(4);
    obs5 << 6617, 1071, 526, 386;
    Eigen::VectorXd obs6(4);
    obs6 << 6188, 1287, 640, 485;
    Eigen::VectorXd obs7(4);
    obs7 << 5681, 1519, 813, 587;
    Eigen::VectorXd obs8(4);
    obs8 << 5116, 1769, 983, 732;
    Eigen::VectorXd obs9(4);
    obs9 << 4473, 2013, 1190, 924;
    Eigen::VectorXd obs10(4);
    obs10 << 3857, 2210, 1404, 1129;

    observations.add_time_point(0.) = obs0;
    observations.add_time_point(1.) = obs1;
    observations.add_time_point(2.) = obs2;
    observations.add_time_point(3.) = obs3;
    observations.add_time_point(4.) = obs4;
    observations.add_time_point(5.) = obs5;
    observations.add_time_point(6.) = obs6;
    observations.add_time_point(7.) = obs7;
    observations.add_time_point(8.) = obs8;
    observations.add_time_point(9.) = obs9;
    observations.add_time_point(10.) = obs10;

    double like = likelihood.compute(observations);
    std::cout << "like" << like << std::endl;
    std::cout << "expected like" << expectedLikelihood << std::endl;
}
