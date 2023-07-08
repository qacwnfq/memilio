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

#include <chrono>

TEST(TestCompartmentLikelihood, computeLikelihoodOseir)
{
    double expectedLikelihood = 484.11127970230297;

    mio::oseir::Model model;
    model.parameters.set<mio::oseir::TimeExposed>(5.2);
    model.parameters.set<mio::oseir::TimeInfected>(6);
    model.parameters.set<mio::oseir::TransmissionProbabilityOnContact>(0.25);
    model.parameters.get<mio::oseir::ContactPatterns>().get_baseline()(0, 0) = 1;

    double total_population                                                                            = 8e8+1e5+1e5+1e3;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Exposed)}]   = 1e5;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Infected)}]  = 1e5;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Recovered)}] = 1e3;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Susceptible)}] =
        total_population -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Exposed)}] -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Infected)}] -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Recovered)}];

    mio::Likelihood<decltype(model)> likelihood(model);

    // random observations were simulated using stochastic differential equations and pyross
    mio::TimeSeries<double> observations(4);
    Eigen::VectorXd obs0(4);
    obs0 << 800000000, 100000, 100000, 1000;
    Eigen::VectorXd obs1(4);
    obs1 << 799974430, 105436, 103089, 18045;
    Eigen::VectorXd obs2(4);
    obs2 << 799947800, 110548, 106801, 35851;
    Eigen::VectorXd obs3(4);
    obs3 << 799921183, 115925, 110623, 53269;
    Eigen::VectorXd obs4(4);
    obs4 << 799891616, 121706, 114866, 72812;
    Eigen::VectorXd obs5(4);
    obs5 << 799862092, 126753, 119617, 92538;
    Eigen::VectorXd obs6(4);
    obs6 << 799832577, 132015, 123923, 112485;
    Eigen::VectorXd obs7(4);
    obs7 << 799799748, 137970, 128836, 134446;
    Eigen::VectorXd obs8(4);
    obs8 << 799767297, 143827, 133673, 156203;
    Eigen::VectorXd obs9(4);
    obs9 << 799733809, 149548, 138980, 178663;
    Eigen::VectorXd obs10(4);
    obs10 << 799700308, 155326, 144198, 201168;

    observations.add_time_point(0.)  = obs0;
    observations.add_time_point(1.)  = obs1;
    observations.add_time_point(2.)  = obs2;
    observations.add_time_point(3.)  = obs3;
    observations.add_time_point(4.)  = obs4;
    observations.add_time_point(5.)  = obs5;
    observations.add_time_point(6.)  = obs6;
    observations.add_time_point(7.)  = obs7;
    observations.add_time_point(8.)  = obs8;
    observations.add_time_point(9.)  = obs9;
    observations.add_time_point(10.) = obs10;

    double actualLikelihood = likelihood.compute(observations);
    EXPECT_DOUBLE_EQ(expectedLikelihood, actualLikelihood);
}

TEST(TestCompartmentLikelihood, benchmark)
{
//    double expectedLikelihood = 484.27044719538532;

    mio::oseir::Model model;
    model.parameters.set<mio::oseir::TimeExposed>(5.2);
    model.parameters.set<mio::oseir::TimeInfected>(6);
    model.parameters.set<mio::oseir::TransmissionProbabilityOnContact>(0.25);
    model.parameters.get<mio::oseir::ContactPatterns>().get_baseline()(0, 0) = 1;

    double total_population                                                                            = 8e8+1e5+1e5+1e3;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Exposed)}]   = 1e5;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Infected)}]  = 1e5;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Recovered)}] = 1e3;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Susceptible)}] =
        total_population -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Exposed)}] -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Infected)}] -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Recovered)}];

    mio::Likelihood<decltype(model)> likelihood(model);

    // random observations were simulated using stochastic differential equations and pyross
    mio::TimeSeries<double> observations(4);
    Eigen::VectorXd obs0(4);
    obs0 << 800000000, 100000, 100000, 1000;
    Eigen::VectorXd obs1(4);
    obs1 << 799974430, 105436, 103089, 18045;
    Eigen::VectorXd obs2(4);
    obs2 << 799947800, 110548, 106801, 35851;
    Eigen::VectorXd obs3(4);
    obs3 << 799921183, 115925, 110623, 53269;
    Eigen::VectorXd obs4(4);
    obs4 << 799891616, 121706, 114866, 72812;
    Eigen::VectorXd obs5(4);
    obs5 << 799862092, 126753, 119617, 92538;
    Eigen::VectorXd obs6(4);
    obs6 << 799832577, 132015, 123923, 112485;
    Eigen::VectorXd obs7(4);
    obs7 << 799799748, 137970, 128836, 134446;
    Eigen::VectorXd obs8(4);
    obs8 << 799767297, 143827, 133673, 156203;
    Eigen::VectorXd obs9(4);
    obs9 << 799733809, 149548, 138980, 178663;
    Eigen::VectorXd obs10(4);
    obs10 << 799700308, 155326, 144198, 201168;

    observations.add_time_point(0.)  = obs0;
    observations.add_time_point(1.)  = obs1;
    observations.add_time_point(2.)  = obs2;
    observations.add_time_point(3.)  = obs3;
    observations.add_time_point(4.)  = obs4;
    observations.add_time_point(5.)  = obs5;
    observations.add_time_point(6.)  = obs6;
    observations.add_time_point(7.)  = obs7;
    observations.add_time_point(8.)  = obs8;
    observations.add_time_point(9.)  = obs9;
    observations.add_time_point(10.) = obs10;

    auto t1 = std::chrono::high_resolution_clock::now();
    volatile double writeMe = 0;
    for(int i=0; i<10; ++i) {
        writeMe = likelihood.compute(observations);
        std::cout << "likelihood: " << writeMe << "\n";
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
}
