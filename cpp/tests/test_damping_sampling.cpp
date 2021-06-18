#include "epidemiology/secir/damping_sampling.h"
#include "epidemiology/secir/contact_matrix.h"
#include "matchers.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cmath>

TEST(TestDampingSampling, apply)
{
    auto ds  = std::vector<epi::DampingSampling>{epi::DampingSampling{epi::UncertainValue(0.5),
                                                                     epi::DampingLevel(0),
                                                                     epi::DampingType(0),
                                                                     epi::SimulationTime(0.0),
                                                                     {0},
                                                                     Eigen::VectorXd::Constant(2, 1.0)},
                                                epi::DampingSampling{epi::UncertainValue(0.25),
                                                                     epi::DampingLevel(1),
                                                                     epi::DampingType(0),
                                                                     epi::SimulationTime(1.0),
                                                                     {
                                                                         0,
                                                                         1,
                                                                     },
                                                                     Eigen::VectorXd::Constant(2, 1.0)}};
    auto cmg = epi::ContactMatrixGroup(2, 2);

    epi::apply_dampings(cmg, ds, [](auto&& v) {
        return epi::make_contact_damping_matrix(v);
    });

    ASSERT_THAT(cmg[0].get_dampings(),
                testing::ElementsAre(epi::SquareDamping(Eigen::MatrixXd::Constant(2, 2, 0.5), epi::DampingLevel(0),
                                                        epi::DampingType(0), epi::SimulationTime(0.0)),
                                     epi::SquareDamping(Eigen::MatrixXd::Constant(2, 2, 0.25), epi::DampingLevel(1),
                                                        epi::DampingType(0), epi::SimulationTime(1.0))));
    ASSERT_THAT(cmg[1].get_dampings(),
                testing::ElementsAre(epi::SquareDamping(Eigen::MatrixXd::Constant(2, 2, 0.25), epi::DampingLevel(1),
                                                        epi::DampingType(0), epi::SimulationTime(1.0))));
}

TEST(TestDampingSampling, contactMask)
{
    auto m = epi::make_contact_damping_matrix((Eigen::VectorXd(2) << 0.0, 0.5).finished()).eval();
    ASSERT_THAT(print_wrap(m), MatrixNear((Eigen::MatrixXd(2, 2) << 0.0, 1-sqrt(0.5), 1-sqrt(0.5), 0.5).finished()));
}

TEST(TestDampingSampling, migrationMask)
{
    auto m = epi::make_migration_damping_vector(epi::ColumnVectorShape(6),
                                                       (Eigen::VectorXd(2) << 0.5, 0.25).finished())
                 .eval();
    ASSERT_THAT(print_wrap(m), MatrixNear((Eigen::VectorXd(6) << 0.5, 0.5, 0.5, 0.25, 0.25, 0.25).finished()));
}