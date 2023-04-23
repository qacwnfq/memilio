/* 
* Copyright (C) 2020-2023 German Aerospace Center (DLR-SC)
*
* Authors: Daniel Abele, Jan Kleinert, Martin J. Kuehn
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
#ifndef SEIR_MODEL_H
#define SEIR_MODEL_H

#include "memilio/compartments/compartmentalmodel.h"
#include "memilio/epidemiology/populations.h"
#include "memilio/epidemiology/contact_matrix.h"
#include "ode_seir/infection_state.h"
#include "ode_seir/parameters.h"

namespace mio
{
namespace oseir
{

/********************
    * define the model *
    ********************/

class Model : public CompartmentalModel<InfectionState, Populations<InfectionState>, Parameters>
{
    using Base = CompartmentalModel<InfectionState, mio::Populations<InfectionState>, Parameters>;

public:
    Model()
        : Base(Populations({InfectionState::Count}, 0.), ParameterSet())
    {
        auto num_compartments                     = this->populations.get_num_compartments();
        Eigen::VectorXd StoE                      = Eigen::VectorXd::Zero(num_compartments);
        StoE[(size_t)InfectionState::Susceptible] = -1;
        StoE[(size_t)InfectionState::Exposed]     = 1;

        Eigen::VectorXd EtoI                   = Eigen::VectorXd::Zero(num_compartments);
        EtoI[(size_t)InfectionState::Exposed]  = -1;
        EtoI[(size_t)InfectionState::Infected] = 1;

        Eigen::VectorXd ItoR                    = Eigen::VectorXd::Zero(num_compartments);
        ItoR[(size_t)InfectionState::Infected]  = -1;
        ItoR[(size_t)InfectionState::Recovered] = 1;

        transition_vectors = Eigen::MatrixXd(num_compartments, 3);
        transition_vectors << StoE, EtoI, ItoR;

        transition_rates = Eigen::VectorXd(transition_vectors.cols());
    }

    void update_transition_rates(Eigen::Ref<const Eigen::VectorXd> pop, Eigen::Ref<const Eigen::VectorXd> y,
                                 double t) const
    {
        auto& params     = this->parameters;
        double coeffStoE = params.get<ContactPatterns>().get_matrix_at(t)(0, 0) *
                           params.get<TransmissionProbabilityOnContact>() / populations.get_total();

        // infection term: proportional to S*I
        transition_rates(0) =
            coeffStoE * y[(size_t)InfectionState::Susceptible] * pop[(size_t)InfectionState::Infected];
        // linear term: proportional to E
        transition_rates(1) =
            1.0 / params.get<TimeExposed>() * y[(size_t)InfectionState::Exposed];
        // linear term: proportional to I
        transition_rates(2) =
            1.0 / params.get<TimeInfected>() * y[(size_t)InfectionState::Infected];
    }

//    /*
//     * @brief updates transition rates but uses relative compartments
//     */
//    void update_transition_rates_relative(Eigen::Ref<const Eigen::VectorXd> pop, Eigen::Ref<const Eigen::VectorXd> y,
//                                          double t) const
//    {
//        auto& params     = this->parameters;
//        double coeffStoE = params.get<ContactPatterns>().get_matrix_at(t)(0, 0) *
//                           params.get<TransmissionProbabilityOnContact>() / populations.get_total();
//
//        // infection term: proportional to S*I
//        transition_rates(0) = coeffStoE * y[(size_t)InfectionState::Susceptible] / populations.get_total() *
//                              pop[(size_t)InfectionState::Infected] / populations.get_total();
//        // linear term: proportional to E
//        transition_rates(1) =
//            1.0 / params.get<TimeExposed>() * y[(size_t)InfectionState::Exposed] / populations.get_total();
//        // linear term: proportional to I
//        transition_rates(2) =
//            1.0 / params.get<TimeInfected>() * y[(size_t)InfectionState::Infected] / populations.get_total();
//    }

    void get_derivatives(Eigen::Ref<const Eigen::VectorXd> pop, Eigen::Ref<const Eigen::VectorXd> y, double t,
                         Eigen::Ref<Eigen::VectorXd> dydt) const override
    {
        //        dydt[(size_t)InfectionState::Susceptible] =
        //            -coeffStoE * y[(size_t)InfectionState::Susceptible] * pop[(size_t)InfectionState::Infected];
        //        dydt[(size_t)InfectionState::Exposed] =
        //            coeffStoE * y[(size_t)InfectionState::Susceptible] * pop[(size_t)InfectionState::Infected] -
        //            (1.0 / params.get<TimeExposed>()) * y[(size_t)InfectionState::Exposed];
        //        dydt[(size_t)InfectionState::Infected] =
        //            (1.0 / params.get<TimeExposed>()) * y[(size_t)InfectionState::Exposed] -
        //            (1.0 / params.get<TimeInfected>()) * y[(size_t)InfectionState::Infected];
        //        dydt[(size_t)InfectionState::Recovered] =
        //            (1.0 / params.get<TimeInfected>()) * y[(size_t)InfectionState::Infected];
        //        dydt = transition_vectors.col(0) * coeffStoE * y[(size_t)InfectionState::Susceptible] *
        //               pop[(size_t)InfectionState::Infected];
        //        dydt += (transition_vectors.col(1) * 1.0 / params.get<TimeExposed>()) * y[(size_t)InfectionState::Exposed];
        //        dydt += (transition_vectors.col(2) * 1.0 / params.get<TimeInfected>()) * y[(size_t)InfectionState::Infected];
        this->update_transition_rates(pop, y, t);
        dydt = transition_vectors * transition_rates;
    }

    void get_noise_correlation(Eigen::Ref<const Eigen::VectorXd> pop, Eigen::Ref<const Eigen::VectorXd> y, double t,
                               Eigen::Ref<Eigen::MatrixXd> noise_correlation) const override
    {
        this->update_transition_rates(pop, y, t);
        Eigen::MatrixXd scaled_transition_vectors = transition_vectors;
        for (long i = 0; i < scaled_transition_vectors.cols(); ++i) {
            scaled_transition_vectors.col(i) *= this->transition_rates(i);
        }
        noise_correlation = scaled_transition_vectors * transition_vectors.transpose();
    }

    void get_drift(Eigen::Ref<const Eigen::VectorXd> pop, Eigen::Ref<const Eigen::VectorXd> y, double t,
                   Eigen::Ref<Eigen::MatrixXd> drift) const override
    {
        auto& params     = this->parameters;
        double coeffStoE = params.get<ContactPatterns>().get_matrix_at(t)(0, 0) *
                           params.get<TransmissionProbabilityOnContact>() / populations.get_total();

        update_transition_rates(pop, y, t);
        Eigen::VectorXd dtransition_0_dx(y.rows());
        dtransition_0_dx << coeffStoE * pop[2] / populations.get_total(), 0,
            coeffStoE * pop[0] / populations.get_total(), 0;
        Eigen::VectorXd dtransition_1_dx(y.rows());
        dtransition_1_dx << 0, 1. / params.get<TimeExposed>(), 0, 0;
        Eigen::VectorXd dtransition_2_dx(y.rows());
        dtransition_2_dx << 0, 0, 1. / params.get<TimeInfected>(), 0;

        drift = transition_vectors.col(0) * dtransition_0_dx.transpose() +
                transition_vectors.col(1) * dtransition_1_dx.transpose() +
                transition_vectors.col(2) * dtransition_2_dx.transpose();
    }

private:
    mutable Eigen::MatrixXd transition_vectors;
    mutable Eigen::VectorXd transition_rates;
    mutable Eigen::MatrixXd dtransition_ratesdY;
};

} // namespace oseir
} // namespace mio

#endif // SEIR_MODEL_H
