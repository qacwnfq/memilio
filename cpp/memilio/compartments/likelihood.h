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
#ifndef MEMILIO_LIKELIHOOD_H
#define MEMILIO_LIKELIHOOD_H

#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"

namespace mio
{

/**
 * @brief A class for the likelihood of a compartment model. Based on https://doi.org/10.1098/rsos.211065.
 * @tparam M a CompartmentModel type
 */
template <class M>
class Likelihood
{
    static_assert(is_compartment_model<M>::value, "Template parameter must be a compartment model.");

public:
    using Model = M;

    /**
     * @brief setup the simulation with an ODE solver
     * @param[in] model: An instance of a compartmental model
     * @param[in] t0 start time
     * @param[in] dt initial step size of integration
     */
    explicit Likelihood(Model const& model)
        : m_integratorCore(
              std::make_shared<mio::ControlledStepperWrapper<boost::numeric::odeint::runge_kutta_cash_karp54>>())
        , m_model(std::make_unique<Model>(model))
    {
    }

    auto compute(TimeSeries<double> observed, double dt = 0.1)
    {
        double logp = 0.;
        for (auto i = 0; i < observed.get_num_time_points() - 1; ++i) {
            double current_time           = observed.get_time(i);
            double next_time              = observed.get_time(i + 1);
            Eigen::VectorXd current_state = observed.get_value(i);
            Eigen::VectorXd next_state    = observed.get_value(i + 1);

            OdeIntegrator integrator(
                [&model = *m_model](auto&& y, auto&& t, auto&& dydt) {
                    model.eval_right_hand_side(y, y, t, dydt);
                },
                current_time, current_state, dt, m_integratorCore);
            integrator.set_integrator(m_integratorCore);
            integrator.advance(next_time);
            Eigen::VectorXd next_simulated =
                interpolate_simulation_result(integrator.get_result(), std::vector<double>{next_time}).get_value(0);

            Eigen::MatrixXd cov = estimate_cond_cov(next_simulated, current_time, next_time, dt);
            Eigen::VectorXd rel_dev = (next_simulated - next_state) / m_model->populations.get_total();

            logp -= rel_dev * rel_dev / (2*m_model->populations.get_total);
        }
        return logp;
    }

    template <class Model>
    Eigen::MatrixXd estimate_cond_cov(Eigen::Ref<Eigen::VectorXd> x, double t1, double t2, double dt)
    {
        auto num_comp          = m_model->populations.get_num_compartments();
        Eigen::MatrixXd sigma0 = Eigen::MatrixXd::Zero(num_comp, num_comp);
        Eigen::MatrixXd drift(num_comp, num_comp);
        Eigen::MatrixXd noise_correlation(num_comp, num_comp);

        Eigen::Map<Eigen::VectorXd> sigma0_vec(sigma0.data(), sigma0.rows() + sigma0.cols(), 1);

        OdeIntegrator integrator(
            [&model = *m_model, &x, &drift, &noise_correlation](auto&& y, auto&& t, auto&& dydt) {
                model.eval_right_hand_side(y, y, t, dydt);
                model.get_drift(x, x, t, drift);
                model.get_noise_correlation(x, x, t, drift);

                Eigen::Map<Eigen::MatrixXd> sig(y.data, drift.rows(), drift.cols());
                // sig is symmetric
                Eigen::MatrixXd dsig_dt = sig * drift.transpose() + drift * sig + noise_correlation;

                Eigen::Map<Eigen::VectorXd> dsig_dt_vec(dsig_dt.data(), dsig_dt.rows() + dsig_dt.cols(), 1);
                return dsig_dt_vec;
            },
            t1, sigma0_vec, dt, m_integratorCore);

        integrator.set_integrator(m_integratorCore);
        integrator.advance(t2);
        Eigen::VectorXd sig_vec =
            interpolate_simulation_result(integrator.get_result(), std::vector<double>{t2}).get_value(0);

        return Eigen::Map<Eigen::MatrixXd>(sig_vec.data(), num_comp, num_comp);
    }

    /**
     * @brief get_integrator
     * @return reference to the core integrator used in the simulation
     */
    IntegratorCore& get_integrator()
    {
        return *m_integratorCore;
    }

    /**
     * @brief returns the simulation model used in simulation
     */
    const Model& get_model() const
    {
        return *m_model;
    }

    /**
     * @brief returns the simulation model used in simulation
     */
    Model& get_model()
    {
        return *m_model;
    }

private:
    std::shared_ptr<IntegratorCore> m_integratorCore;
    std::unique_ptr<Model> m_model;
}; // namespace mio

/**
 * Defines the return type of the `advance` member function of a type.
 * Template is invalid if this member function does not exist.
 * @tparam Sim a compartment model simulation type.
 */
template <class Sim>
using advance_expr_t = decltype(std::declval<Sim>().advance(std::declval<double>()));

/**
 * Template meta function to check if a type is a compartment model simulation.
 * Defines a static constant of name `value`.
 * The constant `value` will be equal to true if Sim is a valid compartment simulation type.
 * Otherwise, `value` will be equal to false.
 * @tparam Sim a type that may or may not be a compartment model simulation.
 */
template <class Sim>
using is_compartment_model_simulation =
    std::integral_constant<bool, (is_expression_valid<advance_expr_t, Sim>::value &&
                                  is_compartment_model<typename Sim::Model>::value)>;

} // namespace mio

#endif //MEMILIO_LIKELIHOOD_H
