// -*- mode: c++ -*-

/*
 * MIT License (MIT)
 *
 * Copyright (c) 2014 Hidekazu Ikeno
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

/**
 * \file   wigner_misc.hpp
 * \brief  Set of miscilialy funtions.for calculating Wigner-D matrix
 */

#ifndef WIGNER_MISC_HPP
#define WIGNER_MISC_HPP

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>

namespace wigner
{

///
/// Meta function for extracting real number type of a scalar type `T`.
///
/// This class provide typedef `type` which is same as `T` if it is a arithmetic
/// number type (typically, a floating point type). If T is a complex number,
/// i.e., T is std::complex<U>, the `type` is same as U.
///

template <typename T>
struct RealTypeOf
{
    using type = T;
};

template <typename T>
struct RealTypeOf<std::complex<T>>
{
    using type = T;
};

///
/// Meta function for cheking whether T is a complex number type.
///
/// It provides static constant member `value` which is equal to `true`, if T is
/// std::complex<U>. For any other type, `value` is `false`.
///
template <typename T>
struct IsComplex
{
    static const bool value = false;
};

template <typename T>
struct IsComplex<std::complex<T>>
{
    static const bool value = true;
};

///
/// Compute complex conjugate of a scalar `x`.
///
template <typename T>
inline T conj(T x)
{
    return x;
}

template <typename T>
inline std::complex<T> conj(const std::complex<T>& x)
{
    return std::conj(x);
}

namespace detail
{

///
/// Adjust the values of $$ \sin\theta $$ and $$ \cos\theta $$ following the
/// Gooding & Wagner's method, so as to make
/// $$\cos^{2}(theta)+\sin^{2}(theta)=1$$ as accuartely as possible.
///
/// Note that This function is used for improve the accuarcy of Wigner D-matrix
/// and related functions.
///
/// @tparam T  type of argument (a floating point type)
///
/// @param[in,out] cs  value of $$\cos\theta$$
/// @param[in,out] sn  value of $$\sin\theta$$
///
/// ### Reference
///
///  R. H. Gooding and C. A. Wagner, "On a Fortran procedure for rotating
///  spherical-harmonic coefficients", Celest. Mech. Dyn. Astr. **108** (2010)
///  pp. 95-106.
///
template <typename T>
void adjust_cos_sin_arguments(T& cs, T& sn)
{
    if (std::abs(sn) < std::abs(cs))
    {
        const T val = std::sqrt((T(1) - cs) * (T(1) + cs));
        sn          = sn < T() ? -val : val;
    }
    else
    {
        const T val = std::sqrt((T(1) - sn) * (T(1) + sn));
        cs          = cs < T() ? -val : val;
    }
}

///
/// @return position index of (l, m) component of spherical harmonics or related
/// functions stored in one-dimentional sequence.
///
/// @param l  degree (angular momentum) of function
/// @param m  order (magnetic moment) of fuction
///
/// @pre  $$l \geq 0` and  $$|m| \leq l$$ must be hold.
///
template <typename IndexT>
IndexT lm_index(int l, int m)
{
    assert(l >= 0);
    assert(std::abs(m) <= l);

    return static_cast<IndexT>(l * (l + 1) + m);
}

} // namespace: detail
} // namespace: wigner

#endif /* WIGNER_MISC_HPP */
