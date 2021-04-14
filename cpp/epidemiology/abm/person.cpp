#include "epidemiology/abm/person.h"
#include "epidemiology/abm/world.h"
#include "epidemiology/abm/location.h"
#include "epidemiology/utils/random_number_generator.h"

namespace epi
{

Person::Person(Location& location, InfectionState state, Index<AbmAgeGroup> age)
    : m_location(location)
    , m_state(state)
    , m_age(age)
{
}

void Person::interact(double dt, const GlobalInfectionParameters& global_infection_params)
{
    auto state     = m_state;
    auto new_state = state;

    if (state == InfectionState::Exposed) {
        if (m_time_until_carrier <= 0) {
            new_state = InfectionState::Carrier;
        }
        m_time_until_carrier -= dt;
    }
    else {
        new_state = m_location.get().interact(*this, dt, global_infection_params);
        if (new_state == InfectionState::Exposed) {
            m_time_until_carrier = global_infection_params.get<IncubationPeriod>()[this->m_age];
        }
    }

    m_state = new_state;
    if (state != new_state) {
        m_location.get().changed_state(*this, state);
    }
}

void Person::migrate_to(Location& location)
{
    m_location.get().remove_person(*this);
    m_location = location;
    m_location.get().add_person(*this);
}
} // namespace epi
