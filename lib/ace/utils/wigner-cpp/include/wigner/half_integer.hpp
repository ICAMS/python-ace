/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016-2017 Hidekazu Ikeno
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

///
/// \file  wigner/HalfInteger.hpp
///
/// Header file for half integer types.
///

#ifndef WIGNER_HALFINTEGER_HPP
#define WIGNER_HALFINTEGER_HPP

#include <cassert>
#include <cmath>
#include <cstdlib>

#include <ios>
#include <limits>
#include <type_traits>

namespace wigner
{

//
// Forward declaration of HalfInteger class
//
template <typename IntT = int>
class HalfInteger;

///
/// IsHalfInteger
/// ---------------
///
/// Check whether `T` is a half integer type.
///
template <typename T>
struct IsHalfInteger
{
    static const constexpr bool value = false;
};

template <typename T>
struct IsHalfInteger<HalfInteger<T>>
{
    static const constexpr bool value = true;
};

///
/// ## IntegerTypeOf
///
/// Meta function that extract base integral type for half integer class.
///
/// If `T` is `HalfInteger<U>`, this class provides member tyepdef `type` equal
/// to `U`, otherwise `type` is equal to `T`.
///
template <typename T>
struct IntegerTypeOf
{
    using type = T;
};

template <typename U>
struct IntegerTypeOf<HalfInteger<U>>
{
    using type = U;
};

///
/// ## make_half
///
/// A tag used for dispatch behavior of constructurs in `HalfInteger` class.
///
struct MakeHalfTag
{
};
// Instantiation of `MakeHalfTag`
constexpr auto make_half = MakeHalfTag();

///
/// ## HalfInteger
///
/// Describes an integer or a half-interger number. This class is mainly used
/// for describing angular momentum numbers.
///
/// ### Usage
///
/// ~~~~~~~~~~~~~~{.cpp}
/// HalfInteger<int> a;                // a = 0
/// HalfInteger<int> b(2);             // b = 2
/// HalfInteger<int> c(3, make_half);  // c = 3/2
/// ~~~~~~~~~~~~~~
///
template <typename IntT>
class HalfInteger
{
public:
    using IntegerType = IntT;

    /// Default constructor. the value is initialized by `IntT()`.
    constexpr HalfInteger() : twice_()
    {
    }

    /// Create a half inteter equals to the given integer value.
    template <typename I>
    constexpr HalfInteger(
        I x, typename std::enable_if<std::is_integral<I>::value>::type* = 0)
        : twice_(2 * static_cast<IntegerType>(x))
    {
    }

    /// Create a half inteter from the given floating point value.
    constexpr HalfInteger(double x) : twice_(get_twice_from_double(x))
    {
    }

    /// Create a half inteter equals to the half of given integer value, `x/2`.
    template <typename I>
    constexpr HalfInteger(
        I x, MakeHalfTag,
        typename std::enable_if<std::is_integral<I>::value>::type* = 0)
        : twice_(static_cast<IntegerType>(x))
    {
    }

    /// Copy constructor.
    constexpr HalfInteger(const HalfInteger&) = default;

    /// Move constructor.
    constexpr HalfInteger(HalfInteger&&) = default;

    /// Copy assignment operator.
    HalfInteger& operator=(const HalfInteger&) = default;

    /// Move assignment operator.
    HalfInteger& operator=(HalfInteger&&) = default;

    /// Assignment operator for integal type.
    template <typename I>
    typename std::enable_if<std::is_integral<I>::value, HalfInteger&>::type
    operator=(I x)
    {
        twice_ = 2 * static_cast<IntegerType>(x);
        return *this;
    }

    /// Assignment operator for floating point type.
    HalfInteger& operator=(double x)
    {
        twice_ = get_twice_from_double(x);
        return *this;
    }

    template <typename T>
    constexpr T castTo() const
    {
        return static_cast<T>(twice()) / static_cast<T>(2);
    }

    void setHalfOf(IntegerType x)
    {
        twice_ = x;
    }

    constexpr IntegerType twice() const
    {
        return twice_;
    }

    constexpr HalfInteger abs() const
    {
        return HalfInteger(std::abs(twice_), make_half);
    }

    constexpr HalfInteger negate() const
    {
        return HalfInteger(-twice_, make_half);
    }

    constexpr bool isHalfInteger() const
    {
        return twice_ % 2 == 1;
    }

    constexpr bool isInteger() const
    {
        return twice_ % 2 == 0;
    }

    // Increment/Decrement
    HalfInteger& operator++()
    {
        twice_ += 2;
        return *this;
    }

    HalfInteger operator++(int)
    {
        HalfInteger result(*this);
        twice_ += 2;
        return result;
    }

    HalfInteger& operator--()
    {
        twice_ -= 2;
        return *this;
    }

    HalfInteger operator--(int)
    {
        HalfInteger result(*this);
        twice_ -= 2;
        return result;
    }

    // operator not
    constexpr bool operator!() const
    {
        return !twice_;
    }

    // boolean conversion
    constexpr operator bool() const
    {
        return twice_;
    }

    //
    // Unary operator
    //
    constexpr HalfInteger operator+() const
    {
        return HalfInteger(*this);
    }

    constexpr HalfInteger operator-() const
    {
        return negate();
    }

    // Arithmetic operators
    template <typename I>
    HalfInteger& operator+=(const HalfInteger<I>& rhs)
    {
        twice_ += rhs.twice();
        return *this;
    }

    template <typename I>
    HalfInteger& operator-=(const HalfInteger<I>& rhs)
    {
        twice_ -= rhs.twice();
        return *this;
    }

    HalfInteger& operator+=(double x)
    {
        twice_ += get_twice_from_double(x);
        return *this;
    }

    HalfInteger& operator-=(double x)
    {
        twice_ -= get_twice_from_double(x);
        return *this;
    }

    static constexpr HalfInteger max()
    {
        return HalfInteger(std::numeric_limits<IntegerType>::max(), make_half);
    }

    static constexpr HalfInteger min()
    {
        return std::is_signed<IntegerType>::value
                   ? HalfInteger(-std::numeric_limits<IntegerType>::max(),
                                 make_half)
                   : HalfInteger(std::numeric_limits<IntegerType>::min(),
                                 make_half);
    }

private:
    static constexpr IntegerType get_twice_from_double(double x)
    {
        return static_cast<IntegerType>(std::floor(2 * x + 0.5));
    }

    IntegerType twice_;
};

///
/// operator==,!=,<,>,<=,>=
/// ========================
///
/// Binary comparison operators for `HalfInteger` type.
///
template <typename I1, typename I2>
inline constexpr bool operator==(const HalfInteger<I1>& lhs,
                                 const HalfInteger<I2>& rhs)
{
    return lhs.twice() == rhs.twice();
}

template <typename I1, typename I2>
inline constexpr bool operator!=(const HalfInteger<I1>& lhs,
                                 const HalfInteger<I2>& rhs)
{
    return !(lhs.twice() == rhs.twice());
}

template <typename I1, typename I2>
inline constexpr bool operator<(const HalfInteger<I1>& lhs,
                                const HalfInteger<I2>& rhs)
{
    return lhs.twice() < rhs.twice();
}

template <typename I1, typename I2>
inline constexpr bool operator>(const HalfInteger<I1>& lhs,
                                const HalfInteger<I2>& rhs)
{
    return (rhs < lhs);
}

template <typename I1, typename I2>
inline constexpr bool operator<=(const HalfInteger<I1>& lhs,
                                 const HalfInteger<I2>& rhs)
{
    return !(rhs < lhs);
}

template <typename I1, typename I2>
inline constexpr bool operator>=(const HalfInteger<I1>& lhs,
                                 const HalfInteger<I2>& rhs)
{
    return !(lhs < rhs);
}

///
/// operator+,-
/// ============
///
/// Binary arithmetic operators for addtion and subtraction between two
/// half integers and those between integer and half integer.
///
template <typename I1, typename I2>
inline constexpr auto operator+(const HalfInteger<I1>& lhs,
                                const HalfInteger<I2>& rhs)
    -> HalfInteger<decltype(I1() + I2())>
{
    using result_type = HalfInteger<decltype(I1() + I2())>;
    return result_type(lhs.twice() + rhs.twice(), make_half);
}

template <typename I1, typename I2>
inline constexpr auto operator+(const HalfInteger<I1>& lhs, I2 rhs)
    -> HalfInteger<decltype(I1() + I2())>
{
    using result_type = HalfInteger<decltype(I1() + I2())>;
    return result_type(lhs.twice() + 2 * rhs, make_half);
}

template <typename I1, typename I2>
inline constexpr auto operator+(I1 lhs, const HalfInteger<I2>& rhs)
    -> HalfInteger<decltype(std::declval<I1>() + std::declval<I2>())>
{
    using result_type = HalfInteger<decltype(I1() + I2())>;
    return result_type(2 * lhs + rhs.twice(), make_half);
}

template <typename I>
inline constexpr HalfInteger<I> operator+(const HalfInteger<I>& lhs, double rhs)
{
    return lhs + HalfInteger<I>(rhs);
}

template <typename I>
inline constexpr HalfInteger<I> operator+(double lhs, const HalfInteger<I>& rhs)
{
    return HalfInteger<I>(lhs) + rhs;
}

template <typename I1, typename I2>
inline constexpr auto operator-(const HalfInteger<I1>& lhs,
                                const HalfInteger<I2>& rhs)
    -> HalfInteger<decltype(I1() - I2())>
{
    using result_type = HalfInteger<decltype(I1() - I2())>;
    return result_type(lhs.twice() - rhs.twice(), make_half);
}

template <typename I1, typename I2>
inline constexpr auto operator-(const HalfInteger<I1>& lhs, I2 rhs)
    -> HalfInteger<decltype(I1() - I2())>
{
    using result_type = HalfInteger<decltype(I1() - I2())>;
    return result_type(lhs.twice() - 2 * rhs, make_half);
}

template <typename I1, typename I2>
inline constexpr auto operator-(I1 lhs, const HalfInteger<I2>& rhs)
    -> HalfInteger<decltype(I1() - I2())>
{
    using result_type = HalfInteger<decltype(I1() - I2())>;
    return result_type(2 * lhs - rhs.twice(), make_half);
}

template <typename I>
inline constexpr HalfInteger<I> operator-(const HalfInteger<I>& lhs, double rhs)
{
    return lhs - HalfInteger<I>(rhs);
}

template <typename I>
inline constexpr HalfInteger<I> operator-(double lhs, const HalfInteger<I>& rhs)
{
    return HalfInteger<I>(lhs) - rhs;
}

///
/// operator<<,>>
/// -------------
///
/// I/O stream operator for `HalfInteger` type. A value of `HalfInteger` is
/// output as a floating point number.
///
template <typename Ch, typename Tr, typename I>
std::basic_istream<Ch, Tr>& operator>>(std::basic_istream<Ch, Tr>& is,
                                       HalfInteger<I>& x)
{
    double v;
    if (is >> v)
    {
        x = v;
    }
    return is;
}

template <typename Ch, typename Tr, typename I>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os,
                                       const HalfInteger<I>& x)
{
    os << x.template castTo<double>();
    return os;
}

///
/// castTo
/// -------
///
/// Cast `x` of `ArgumentT` type to `ResultT` type.
///
template <typename ResultT, typename ArgumentT>
constexpr ResultT castTo(ArgumentT x)
{
    return static_cast<ResultT>(x);
}

template <typename ResultT, typename I>
constexpr ResultT castTo(HalfInteger<I> x)
{
    return x.template castTo<ResultT>();
}

///
/// abs
/// ---
///
/// Compute absolute value of the given argument `x`. The behavior is undefined
/// if the result cannot be represented by the return type.
///
template <typename T>
constexpr T abs(const T& x)
{
    return std::abs(x);
}

template <typename I>
constexpr HalfInteger<I> abs(const HalfInteger<I>& x)
{
    return x.abs();
}

///
/// twice_plus_one
/// --------------
///
/// For given argument `x` with type `T`, the function returns:
///  - `2 * x + 1` if type `T` is an arithmetic type
///  - `x.twice() + 1` if `IsHalfInteger<T>::value` is `true`.
///
template <typename T>
constexpr T twicePlusOne(T x)
{
    return 2 * x + 1;
}

template <typename I>
constexpr I twicePlusOne(const HalfInteger<I>& x)
{
    return x.twice() + 1;
}

///
/// is_integer
/// ----------
///
/// For given argument `x` with type `T`, the function returns:
///  - `true` if `x` is an integer
///  - `false` if `x` is a half-integer
///
template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, bool>::type
    isInteger(T /* x */)
{
    return true;
}

template <typename I>
constexpr bool isInteger(const HalfInteger<I>& x)
{
    return x.is_integer();
}

} // namespace: wigner

#endif /* WIGNER_HALF_INTEGER_HPP */
