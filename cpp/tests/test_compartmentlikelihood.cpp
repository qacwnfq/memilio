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
    double expectedLikelihood = 84.13888188888475;

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

    // random observations were simulated using stochastic differential equations and pyross
    mio::TimeSeries<double> observations(4);
    Eigen::VectorXd obs0(4);
    obs0 << 8.000e+03,3.000e+02, 2.000e+02, 1.000e+02;
    Eigen::VectorXd obs1(4);
    obs1 << 7.797e+03,4.350e+02, 2.340e+02, 1.340e+02;
    Eigen::VectorXd obs2(4);
    obs2 << 7.571e+03,5.750e+02, 2.800e+02, 1.740e+02;
    Eigen::VectorXd obs3(4);
    obs3 << 7.296e+03,7.230e+02, 3.630e+02, 2.180e+02;
    Eigen::VectorXd obs4(4);
    obs4 << 6.919e+03,9.440e+02, 4.560e+02, 2.810e+02;
    Eigen::VectorXd obs5(4);
    obs5 << 6.528e+03,1.159e+03, 5.510e+02, 3.620e+02;
    Eigen::VectorXd obs6(4);
    obs6 << 6.066e+03,1.365e+03, 7.050e+02, 4.640e+02;
    Eigen::VectorXd obs7(4);
    obs7 << 5.529e+03, 1.599e+03, 8.880e+02, 5.840e+02;
    Eigen::VectorXd obs8(4);
    obs8 << 4.930e+03, 1.849e+03, 1.050e+03, 7.710e+02;
    Eigen::VectorXd obs9(4);
    obs9 << 4.320e+03, 2.120e+03, 1.193e+03, 9.670e+02;
    Eigen::VectorXd obs10(4);
    obs10 << 3.757e+03, 2.270e+03, 1.402e+03, 1.171e+03;

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
    std::cout << "like " << like << std::endl;
    std::cout << "expected like " << expectedLikelihood << std::endl;
}
