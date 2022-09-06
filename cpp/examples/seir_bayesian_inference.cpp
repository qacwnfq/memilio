/*
* Copyright (C) 2020-2021 German Aerospace Center (DLR-SC)
*
* Authors: Johann Fredrik Jadebeck
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
#include <cmath>
#include <hops/hops.hpp>
#include <utility>

#include "ode_seir/model.h"
#include "ode_seir/infection_state.h"
#include "ode_seir/parameters.h"
#include "memilio/compartments/simulation.h"
#include "memilio/utils/logging.h"


class seir_model : public hops::Model {
public:
    seir_model(double t0,
               double tmax,
               double dt,
               Eigen::VectorXd std,
               mio::TimeSeries<double> observations,
               mio::oseir::Model modelImpl) :
            t0(t0),
            tmax(tmax),
            dt(dt),
            std(std::move(std)),
            observations(std::move(observations)),
            modelImpl(std::move(modelImpl)) {}

    double computeNegativeLogLikelihood(const Eigen::VectorXd &x) override {
        modelImpl.parameters.set<mio::oseir::LatentTime>(x(0));
        modelImpl.parameters.set<mio::oseir::InfectiousTime>(x(1));
        modelImpl.parameters.set<mio::oseir::InfectionProbabilityFromContact>(x(2));
        modelImpl.parameters.get<mio::oseir::ContactPatterns>().get_baseline()(0, 0) = x(3);

        auto simulatedData = simulate(t0, tmax, dt, modelImpl);

        double SSR = 0;
        // TODO how to match times efficiently, when observations are not available for every time?
        for (long i = 0; i < observations.get_num_time_points(); ++i) {
            SSR += ((observations.get_value(i) - simulatedData.get_value(i))
                    .cwiseQuotient(std.cwiseProduct(observations.get_value(i))))
                    .squaredNorm();
        }
        return SSR;
    }

    std::vector<std::string> getDimensionNames() const override {
        return std::vector<std::string>{
                "LatentTime",
                "InfectiousTime",
                "InfectionProbabilityFromContact",
                "ContactPatterns"
        };
    }

    std::unique_ptr<hops::Model> copyModel() const override {
        return std::make_unique<seir_model>(*this);
    }

private:
    double t0;
    double tmax;
    double dt;
    Eigen::VectorXd std;
    mio::TimeSeries<double> observations;
    mio::oseir::Model modelImpl;
};

int main() {
    mio::set_log_level(mio::LogLevel::debug);

    double t0 = 0;
    double tmax = 1;
    double dt = 0.001;

    mio::log_info("Simulating SEIR; t={} ... {} with dt = {}.", t0, tmax, dt);

    mio::oseir::Model model;

    double total_population = 10000;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Exposed)}] = 100;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Infected)}] = 100;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Recovered)}] = 100;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Susceptible)}] =
            total_population -
            model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Exposed)}] -
            model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Infected)}] -
            model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Recovered)}];
    // suscetible now set with every other update
    // params.nb_sus_t0   = params.nb_total_t0 - params.nb_exp_t0 - params.nb_inf_t0 - params.nb_rec_t0;
    double latent_time = 5.2;
    double infectious_time = 6;
    double infection_probability_from_contact = 0.04;
    double contact_patterns = 10;
    model.parameters.set<mio::oseir::LatentTime>(latent_time);
    model.parameters.set<mio::oseir::InfectiousTime>(infectious_time);
    model.parameters.set<mio::oseir::InfectionProbabilityFromContact>(infection_probability_from_contact);
    model.parameters.get<mio::oseir::ContactPatterns>().get_baseline()(0, 0) = contact_patterns;

    auto seir = simulate(t0, tmax, dt, model);

    auto perturbed_seir = seir;

    // relative stds
    double exposed_std = 0.01;
    double infected_std = 0.01;
    double recovered_std = 0.01;
    double susceptible_std = 0.01;
    Eigen::VectorXd std(4);
    std << exposed_std,
            infected_std,
            recovered_std,
            susceptible_std;


    hops::RandomNumberGenerator rng(42);

    for (auto it = seir.begin(); it != seir.end(); ++it) {
        for (long i = 0; i < (*it).rows(); ++i) {
            (*it)[i] += std::normal_distribution<double>(0, (*it)[i] * std(i))(rng);
            (*it)[i] = std::round((*it)[i]);
        }
    }

    auto model_for_mcmc = seir_model(t0, tmax, dt, std, perturbed_seir, model);

    Eigen::MatrixXd A(8, 4);
    A << Eigen::MatrixXd::Identity(4, 4), -Eigen::MatrixXd::Identity(4, 4);
    Eigen::VectorXd b(8);
    b << 15 * Eigen::VectorXd::Ones(4), Eigen::VectorXd::Zero(4);
    Eigen::VectorXd start(4);
    start << latent_time, infectious_time, infection_probability_from_contact, contact_patterns;


    auto markovChain = hops::MarkovChainFactory::createMarkovChain(
            hops::MarkovChainType::CoordinateHitAndRun,
            A,
            b,
            start,
            model_for_mcmc);

    markovChain->setParameter(hops::ProposalParameter::STEP_SIZE, 0.25);

    long n_samples = 10'000;
    std::vector<double> acceptance_rates;
    std::vector<Eigen::VectorXd> states;
    std::vector<double> negative_log_likelihoods;
    for (long i = 0; i < n_samples; ++i) {
        auto[acceptance_rate, state] = markovChain->draw(rng, A.cols());
        negative_log_likelihoods.emplace_back(markovChain->getStateNegativeLogLikelihood());
        acceptance_rates.emplace_back(acceptance_rate);
        states.emplace_back(state);
    }

    auto fileWriter = hops::FileWriterFactory::createFileWriter("seir_samples", hops::FileWriterType::CSV);
    fileWriter->write("states", states);
    fileWriter->write("acceptance_rates", acceptance_rates);
    fileWriter->write("negative_log_likelihoods", negative_log_likelihoods);
    fileWriter->write("parameter_names", model_for_mcmc.getDimensionNames());
}

