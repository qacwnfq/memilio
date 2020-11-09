#ifndef EPI_UTILS_TYPE_SAFE_H
#define EPI_UTILS_TYPE_SAFE_H

#include <ostream>

namespace epi
{

/**
 * typesafe wrapper around any type to make function arguments, tuple elements, etc. easily distinguishable.
 * e.g.
 *   class Foo : public Type<int, Foo>() {};
 *   class Bar : public Type<int, Bar>() {};
 *   void work(Foo f, Bar b);
 * instead of
 *   void work(int f, int b); 
 */
template <class T, class Derived>
class TypeSafe
{
public:
    /**
     * default constructor.
     */
    TypeSafe()
    {
    }

    /**
     * value constructor.
     */
    explicit TypeSafe(T t)
        : m_t(t)
    {
    }

    /**
     * conversion to underlying type.
     */
    explicit operator T() const
    {
        return m_t;
    }

    /**
     * equality operators.
     */
    bool operator==(const Derived& other) const
    {
        return m_t == other.m_t;
    }
    bool operator!=(const Derived& other) const
    {
        return !(*this == other);
    }

    /**
     * stream operators.
     */
    friend std::ostream& operator<<(std::ostream& os, const TypeSafe& ts)
    {
        return (os << ts.m_t);
    }

private:
    T m_t;
};

/**
 * helper macro to declare a class that derives from TypeSafe.
 */
#define DECL_TYPESAFE(Name, T)                                                                                         \
    struct Name : public ::epi::TypeSafe<T, Name> {                                                                    \
        using TypeSafe<T, Name>::TypeSafe;                                                                             \
    }

} // namespace epi

#endif //EPI_UTILS_TYPE_SAFE_H