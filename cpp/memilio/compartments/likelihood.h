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

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/math/constants/constants.hpp>

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

    std::pair<Eigen::VectorXd, double> solve_covariance_inverse(Eigen::Ref<Eigen::MatrixXd> cov,
                                                                Eigen::Ref<Eigen::VectorXd> dev,
                                                                double eps=1e-13)
    {
        Eigen::LLT<Eigen::MatrixXd> llt_solver(cov);
        if (llt_solver.info() == Eigen::Success) {
            Eigen::VectorXd inv_cov_dev = llt_solver.solve(dev);
            double ldet                 = Eigen::MatrixXd(llt_solver.matrixL()).diagonal().array().log().sum();
            return std::make_pair(inv_cov_dev, ldet);
        }

        // If llt does not work numerically, use pseudo-inverse
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
        Eigen::VectorXd es = solver.eigenvalues();
        double ldet        = 0;
        for (Eigen::Index j = 0; j < es.rows(); ++j) {
            if (es(j) < eps) {
                es(j) = 0;
            }
            else {
                ldet += std::log(es(j));
                es(j) = 1. / es(j);
            }
        }
        Eigen::MatrixXd evs         = solver.eigenvectors();
        Eigen::VectorXd inv_cov_dev = (evs * es.asDiagonal() * evs.transpose()) * dev;

        return std::make_pair(inv_cov_dev, ldet/2);
    }

    auto compute(TimeSeries<double> observed, double dt = 1e-1, double eps = 1e-13)
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

            Eigen::MatrixXd cov = this->estimate_cond_cov(integrator.get_result(), current_time, next_time, dt);

            Eigen::VectorXd next_simulated =
                interpolate_simulation_result(integrator.get_result(), std::vector<double>{next_time}).get_value(0);
            Eigen::VectorXd dev = next_state - next_simulated;

            auto result_pair = solve_covariance_inverse(cov, dev, eps);
            Eigen::VectorXd inv_cov_dev = result_pair.first;
            double ldet = result_pair.second;

            double num_comparts   = m_model->populations.get_num_compartments();
            double num_age_groups = 1; // hard coded for now
            double dim            = num_comparts * num_age_groups;
            logp -= static_cast<decltype(logp)>(dev.transpose() * inv_cov_dev) / 2 - ldet;
            logp -= dim / 2 * std::log(2 * boost::math::constants::pi<double>());
        }
        return -logp;
    }

    Eigen::MatrixXd estimate_cond_cov(TimeSeries<double> const& ts, double t1, double t2, double dt)
    {
        auto num_comp = m_model->populations.get_num_compartments();
        Eigen::MatrixXd drift(num_comp, num_comp);
        Eigen::MatrixXd noise_correlation(num_comp, num_comp);
        Eigen::VectorXd sigma0_vec = Eigen::VectorXd::Zero(num_comp * num_comp);

        OdeIntegrator integrator(
            [&model = *m_model, &ts, &drift, &noise_correlation](auto&& y, auto&& t, auto&& dsig_dt_vec) {
                Eigen::VectorXd x = interpolate_simulation_result(ts, std::vector<double>{t}).get_value(0);
                model.get_drift(x, x, t, drift);
                model.get_noise_correlation(x, x, t, noise_correlation);

                const Eigen::MatrixXd sig = Eigen::Map<const Eigen::MatrixXd>(y.data(), drift.rows(), drift.cols());
                // sig is symmetric
                Eigen::MatrixXd dsig_dt = sig * drift.transpose() + drift * sig + noise_correlation;

                dsig_dt_vec = Eigen::Map<Eigen::VectorXd>(dsig_dt.data(), y.rows(), y.cols());
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
};

} // namespace mio

#endif //MEMILIO_LIKELIHOOD_H
