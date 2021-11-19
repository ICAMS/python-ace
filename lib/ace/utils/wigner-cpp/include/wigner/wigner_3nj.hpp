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
/// \file   wigner/wigner_3nj.hpp
///
/// \brief  Wigner 3-j, 6-j, 9-j symbol calculator
///

#ifndef WIGNER_WIGNER_THREE_NJ_HPP
#define WIGNER_WIGNER_THREE_NJ_HPP

#include <cassert>
#include <cmath>
#include <cstdlib>

#include <limits>
#include <vector>
#include <tuple>

#include "half_integer.hpp"

namespace wigner
{

namespace detail
{
//
// ## check_triad
//
// Check whether given three values `(j1, j2, j3)` satisfies triangular
// relations. In case the arguments are half-integer check also whether
// `j1 + j2 + j3` is an integer.
//
// ### Pre-condition
// All arguments, j1, j2, and j3, must be non-negative (>= 0)
//
template <typename I>
inline typename std::enable_if<std::is_integral<I>::value, bool>::type
check_triad(I j1, I j2, I j3)
{
    return abs(j1 - j2) <= j3 && j3 <= j1 + j2;
}

template <typename I>
inline bool check_triad(const HalfInteger<I>& j1, const HalfInteger<I>& j2,
                        const HalfInteger<I>& j3)
{
    return (j1 + j2 + j3).isInteger() && abs(j1 - j2) <= j3 && j3 <= j1 + j2;
}

//
// ## check_jm_pair
//
// Check whether given `m` value is in valid range for given `j`.
// The result is `true` if `abs(m) <= j` and `j + m` is an integer, `false`
// otherwise.
//
template <typename I>
inline typename std::enable_if<std::is_integral<I>::value, bool>::type
check_jm_pair(I j, I m)
{
    return abs(m) <= j;
}

template <typename I>
inline bool check_jm_pair(const HalfInteger<I>& j, const HalfInteger<I>& m)
{
    return (j + m).isInteger() && abs(m) <= j;
}

//
// ## detail::rec_luscombe_luban
//
// This function solves the three-term recurrence relation for Wigner 3-j or 6-j
// symbols deribed by Schulten and Gordon [1], which can be written in general
// as,
//
//   X(n) f(n+1) + Y(n) f(n) + Z(n) f(n-1) = 0  for nmin <= n <= nmax.
//
// The recursive algrithm proposed by J. H. Luscombe and M. Luban [2] is used to
// avoid the overflow during the recursion.
//
// ### References
//
// [1] Klaus Schulten and Roy G. Gordon, "Exact recursive evaluation of 3j and
//     6j-coefficients for quantum-mechanical coupling of angular momenta,"
//     J. Math. Phys. 16, pp 1961-1970 (1975).
//
// [2] James H. Luscombe and Marshall Luban, "Simplified recursive algrithm for
//     Wigner 3j and 6j symbols," Phys. Rev. E 57, pp. 7274-7277 (1998).
//
template <typename ArgT, typename Iterator, typename FuncXYZ>
void rec_luscombe_luban(ArgT nmin_, ArgT nmax_, Iterator dest, const FuncXYZ fn)
{
    using value_t = typename std::iterator_traits<Iterator>::value_type;
    using diff_t  = typename std::iterator_traits<Iterator>::difference_type;
    using std::abs; // for ADL
    //
    // ----- Constants for internal use
    //
    static const auto one = value_t(1);
    static const auto tiny =
        std::sqrt(std::sqrt(std::numeric_limits<value_t>::min()));

    const auto siz = castTo<diff_t>(nmax_ - nmin_ + 1);
    auto nmin      = castTo<value_t>(nmin_);
    auto nmax      = castTo<value_t>(nmax_);

    //--------------------------------------------------------------------------
    // Non-linear two-term recurrence in the non-classical region.
    //--------------------------------------------------------------------------
    //
    // ===== Backward recursion =====
    //
    // Compute the ratios of successive values of f[n], r[n] = f[n] / f[n-1], by
    // the following two-term recurrence relation:
    //
    //   r(nmax) = -Z(nmax) / Y(nmax).
    //   r[n] = -Z(n) / (Y(n) + X(n)r[n+1])  for n2 <= n <= nmax - 1
    //
    // where n2 is the value of n that r[n] first exceeds unity, i.e.,
    //
    //   r[n2] > 1 but r[n2+1] < 1.
    //
    // Then, compute the sequence f[n] by using
    //
    //   f[n2+i] = f[n2] * r[n2+1] * ... * r[n2 + i].
    //
    // For the convenience, we set f[n2] = 1.
    //
    auto i2         = siz - 1;
    auto n2         = nmax;
    const auto ymax = fn.Y(nmax);
    if (abs(ymax) < tiny)
    {
        // Y(nmax) = 0. Just set f(nmax) = 1.
        dest[i2] = one;
    }
    else
    {
        auto rn  = -fn.Z(nmax) / ymax;
        dest[i2] = rn;

        while (--i2 > 0)
        {
            --n2;
            rn = -fn.Z(n2) / (fn.Y(n2) + fn.X(n2) * rn);
            if (abs(rn) > one)
            {
                break;
            }
            dest[i2] = rn;
        }
        //
        // Construct the sequence f[n] (for n2 < n <= nmax) with f[n2] set to
        // unity.
        //
        // Note that in this convension f[n2+1] holds the value of r[n2+1],
        // which is convenient for latter use.
        //
        auto k  = i2;
        dest[k] = one;
        // dest[n2+1] = r[n2+1] = f[n2+1]
        while (++k < siz - 1)
        {
            dest[k + 1] *= dest[k];
        }
    }

    if (i2 == 0)
    {
        return;
    }
    //
    // ===== Forward recursion =====
    //
    // Compute the ratios of successive values of f'[n], s[n] = s[n] / s[n+1],
    // by the following two-term recurrence relation:
    //
    //   s(nmax) = -X(nmin) / Y(nmin).
    //   s[n] = -X(n) / (Y(n) + Z(n)r[n-1])  for nmin + 1 <= n <= min(n1, n2)
    //
    // where n1 is the value of n that s[n] first exceeds unity, i.e.,
    //
    //   s[n1] > 1 but s[n1-1] < 1.
    //
    // Then, compute the sequence f[n] by using
    //
    //   f'[n1-k] = f'[n1] * s[n1-1] * ... * s[n1 - k].
    //
    // For the convenience, we set f'[n1] = 1.
    //
    auto i1         = decltype(siz)();
    auto n1         = nmin;
    const auto ymin = fn.Y(nmin);
    if (abs(ymin) < tiny)
    {
        // Y(nmin) = 0. Just set f(nmin) = 1.
        dest[i1] = one;
    }
    else
    {
        auto sn  = -fn.X(nmin) / ymin;
        dest[i1] = sn;

        while (++i1 < i2)
        {
            ++n1;
            sn = -fn.X(n1) / (fn.Y(n1) + fn.Z(n1) * sn);
            if (abs(sn) > one)
            {
                break;
            }
            dest[i1] = sn;
        }
        //
        // Construct the sequence f'[n] (for n1 < n <= nmax) with f[n1] set to
        // unity.
        //
        // Note that in this convension, f[n1-1] holds the value of s[n1-1],
        // which is convenient for latter use.
        //
        auto k  = i1;
        dest[k] = one;
        while (--k > 0)
        {
            dest[k - 1] *= dest[k];
        }
    }

    if (i1 < i2)
    {
        //----------------------------------------------------------------------
        // Linear three-term recurrence in the classical region.
        //----------------------------------------------------------------------
        //
        // ===== Forward recursion =====
        //
        // Compute the sequence t[n] = f'[n] / f'[n2], by the following
        // three-term recurrence relation:
        //
        //   t[n1-1] = s[n1-1],
        //   t[n1] = 1,
        //   X(n)t[n+1] + Y(n)t[n] + Z(n) t[n-1] = 0   for n1 + 1 <= n <= n2
        //
        // NOTE: Here `ret` holds,
        //
        //   dest[n  - nmin] = f'[n] = t[n]  for nmin <= n < n1
        //   dest[n1 - nmin] = 1
        //   dest[n  - nmin] = undefined     for n1 < n < n2
        //   dest[n2 - nmin] = 1
        //   dest[n  - nmin] = f[n]          for n2 < n <= nmax
        //
        auto n = n1;
        if (i1 == 0)
        {
            dest[i1 + 1] = (-fn.Y(n) * dest[i1]) / fn.X(n);
            ++n;
            ++i1;
        }

        for (auto k = i1; k < i2; ++k)
        {
            const auto numer = -fn.Y(n) * dest[k] - fn.Z(n) * dest[k - 1];
            dest[k + 1]      = numer / fn.X(n);
            ++n;
        }
        //
        // Now `ret` holds
        //   dest[n  - nmin] = t[n]  for nmin <= n <= n2
        //  (ret[n1 - nmin] = 1)
        //   dest[n  - nmin] = f[n]  for n2 < n <= nmax
        //
        // Finally, scalce t[n] by 1 / t[n2] for nmin <= n <= n2 to get the
        // sequence f[n] for all allowed n (nmin <= n <= nmax).
        //
        const auto scale = one / dest[i2];
        for (decltype(i2) k = 0; k <= i2; ++k)
        {
            dest[k] *= scale;
        }
    }

    return;
}

//------------------------------------------------------------------------------
// Implementation for the function `f(j)= wigner3j(j1, j2, j, m1, m2, m)`
//------------------------------------------------------------------------------
//
// Three-term recurrence formula for function
// `f(j)= wigner3j(j1, j2, j, m1, m2, m)`
// where `jmin <= j <= jmax` and `jmin == 0`
//
template <typename T, typename ArgT>
struct funcs_xyz_wigner3j_rec_j_jmin0
{
    funcs_xyz_wigner3j_rec_j_jmin0(ArgT j1, ArgT j2, ArgT m1, ArgT m2)
        : a2(castTo<T>(j1 + j2 + 1)), b1(castTo<T>(m2 - m1))
    {
    }

    T X(T j) const
    {
        return Z(j + 1);
    }

    T Y(T j) const
    {
        return (2 * j + 1) * b1;
    }

    T Z(T j) const
    {
        return j * std::sqrt((a2 + j) * (a2 - j));
    }

private:
    T a2, b1;
};

//
// Three-term recurrence for function
// `f(j)= wigner3j(j1, j2, j, m1, m2, m)`
// where `jmin <= j <= jmax` and `jmin > 0`
//
template <typename T, typename ArgT>
struct funcs_xyz_wigner3j_rec_j
{
    funcs_xyz_wigner3j_rec_j(ArgT j1, ArgT j2, ArgT m1, ArgT m2)
        : a1(castTo<T>(j1 - j2)),
          a2(castTo<T>(j1 + j2 + 1)),
          a3(castTo<T>(-m1 - m2)),
          b1(castTo<T>(m2 - m1)),
          b2(a3 * (castTo<T>(j1) * castTo<T>(j1 + 1) -
                   castTo<T>(j2) * castTo<T>(j2 + 1)))
    {
    }

    T aux(T j) const
    {
        return std::sqrt((a1 + j) * (a1 - j) * (a2 + j) * (a2 - j) * //
                         (a3 - j) * (a3 + j));
    };

    T X(T j) const
    {
        return j * aux(j + 1);
    }

    T Y(T j) const
    {
        return (2 * j + 1) * (j * (j + 1) * b1 - b2);
    }

    T Z(T j) const
    {
        return (j + 1) * aux(j);
    }

private:
    T a1, a2, a3;
    T b1, b2;
};
//
// Implementation body
//
template <typename T, typename ArgT>
struct wigner3j_rec_j_impl
{
    using Scalar       = T;
    using ArgumentType = ArgT;
    using IntegerType  = typename IntegerTypeOf<ArgT>::type;

    static std::tuple<ArgumentType, ArgumentType> args_range(ArgumentType j1,
                                                             ArgumentType j2,
                                                             ArgumentType m1,
                                                             ArgumentType m2)
    {
        if (check_jm_pair(j1, m1) && check_jm_pair(j2, m2))
        {
            const auto jmin = std::max(abs(j1 - j2), abs(m1 + m2));
            const auto jmax = j1 + j2;
            return std::make_tuple(jmin, jmax);
        }
        // All symbols are zero for given arguments. Return negative value
        // pairs satisfying jmin = jmax + 1
        return std::make_tuple(ArgumentType(-1), ArgumentType(-2));
    }

    template <typename SizeT>
    static SizeT storage_size(ArgumentType jmin, ArgumentType jmax,
                              ArgumentType, ArgumentType, ArgumentType,
                              ArgumentType)
    {
        return castTo<SizeT>(jmax - jmin + 1);
    }

    template <typename RandomAccessIterator>
    static void run(RandomAccessIterator out, ArgumentType jmin,
                    ArgumentType jmax, ArgumentType j1, ArgumentType j2,
                    ArgumentType m1, ArgumentType m2)
    {
        using difference_type = typename std::iterator_traits<
            RandomAccessIterator>::difference_type;
        constexpr const Scalar sign[2] = {Scalar(1), Scalar(-1)};

        auto parity = castTo<difference_type>(j1 - j2 + m1 + m2);
        auto jj1    = castTo<Scalar>(twicePlusOne(jmin));

        if (jmin == jmax)
        {
            // Only one allowed symbol.
            *out = sign[parity & 1] / std::sqrt(jj1);
            return;
        }

        if (jmin == 0)
        {
            //
            // This is reached when j1 = j2 and m1 = -m2. In this case, j(j+1)
            // is
            // factor out from X(j), Y(j) and Z(j): otherwise, X(jmin) becomes
            // zero
            // and the recursion fails.
            //
            funcs_xyz_wigner3j_rec_j_jmin0<T, ArgT> xyz(j1, j2, m1, m2);
            rec_luscombe_luban(jmin, jmax, out, xyz);
        }
        else
        {
            funcs_xyz_wigner3j_rec_j<T, ArgT> xyz(j1, j2, m1, m2);
            rec_luscombe_luban(jmin, jmax, out, xyz);
        }
        //
        // Normalization:
        //  \sum_{j=j_{min}}^{j_{max}} (2j+1)f^{2}(j) = 1
        //
        auto sum  = Scalar();
        auto last = out + castTo<difference_type>(jmax - jmin + 1);
        for (auto it = out; it != last; ++it)
        {
            sum += jj1 * (*it) * (*it);
            jj1 += Scalar(2); // fac = 2j + 1
        }
        // Enforce sgn[f(jmax)] == (-1)**parity
        parity += static_cast<difference_type>(*(last - 1) < Scalar());
        const auto scale = sign[parity & 1] / std::sqrt(sum);
        *out *= scale;
        while (++out != last)
        {
            (*out) *= scale;
        }
    }
};

//------------------------------------------------------------------------------
// Implementation for the function `g(m)= wigner3j(j1, j2, j3, m1, m, -m-m1)`
//------------------------------------------------------------------------------
//
// Three-term recurrence for function g(m)= wigner3j(j1, j2, j3, m1, m, -m - m1)
// where `mmin <= m <= mmax`.
//
template <typename T, typename ArgT>
struct funcs_xyz_wigner3j_rec_m
{
    funcs_xyz_wigner3j_rec_m(ArgT j1, ArgT j2, ArgT j3, ArgT m1)
        : c0(castTo<T>(j2)),
          c1(castTo<T>(j3 - m1 + 1)),
          c2(castTo<T>(j3 + m1)),
          d1(castTo<T>(j2) * castTo<T>(j2 + 1) +
             castTo<T>(j3) * castTo<T>(j3 + 1) -
             castTo<T>(j1) * castTo<T>(j1 + 1)),
          d2(castTo<T>(m1))
    {
    }

    T X(T m) const
    {
        return Z(m + 1);
    }

    T Y(T m) const
    {
        return (d1 - 2 * m * (m + d2));
    }

    T Z(T m) const
    {
        return std::sqrt((c0 - m + 1) * (c0 + m) * (c1 - m) * (c2 + m));
    }

private:
    T c0, c1, c2;
    T d1, d2;
};

//
// Implementation body
//
template <typename T, typename ArgT>
struct wigner3j_rec_m_impl
{
    using Scalar       = T;
    using ArgumentType = ArgT;
    using IntegerType  = typename IntegerTypeOf<ArgT>::type;

    static std::tuple<ArgumentType, ArgumentType> args_range(ArgumentType j1,
                                                             ArgumentType j2,
                                                             ArgumentType j3,
                                                             ArgumentType m1)
    {
        if (check_jm_pair(j1, m1) && check_triad(j1, j2, j3))
        {
            const auto mmin = std::max(-j2, -j3 - m1);
            const auto mmax = std::min(j2, j3 - m1);
            return std::make_tuple(mmin, mmax);
        }
        // All symbols are zero for given arguments. Return negative value
        // pairs satisfying jmin = jmax + 1
        return std::make_tuple(ArgumentType(-1), ArgumentType(-2));
    }

    template <typename SizeT>
    static SizeT storage_size(ArgumentType mmin, ArgumentType mmax,
                              ArgumentType, ArgumentType, ArgumentType,
                              ArgumentType)
    {
        return castTo<SizeT>(mmax - mmin + 1);
    }

    template <typename RandomAccessIterator>
    static void run(RandomAccessIterator out, ArgumentType mmin,
                    ArgumentType mmax, ArgumentType j1, ArgumentType j2,
                    ArgumentType j3, ArgumentType m1)
    {
        using difference_type = typename std::iterator_traits<
            RandomAccessIterator>::difference_type;
        constexpr const Scalar sign[2] = {Scalar(1), Scalar(-1)};

        auto parity    = castTo<IntegerType>(j2 - j3 - m1);
        const auto pre = T(1) / std::sqrt(castTo<T>(twicePlusOne(j1)));
        if (mmin == mmax)
        {
            *out = std::copysign(pre, sign[parity & 1]);
            return;
        }

        funcs_xyz_wigner3j_rec_m<T, ArgT> xyz(j1, j2, j3, m1);
        rec_luscombe_luban(mmin, mmax, out, xyz);
        //
        // Normalization
        //
        auto sum  = Scalar();
        auto last = out + castTo<difference_type>(mmax - mmin) + 1;
        for (auto it = out; it != last; ++it)
        {
            sum += (*it) * (*it);
        }

        parity += static_cast<difference_type>(*(last - 1) < T());

        const auto scale =
            std::copysign(pre, sign[parity & 1]) / std::sqrt(sum);

        *out *= scale;
        while (++out != last)
        {
            *out *= scale;
        }
    }
};

//------------------------------------------------------------------------------
// Implementation for the function `h(j) = wigner6j(j, j2, j3, j4, j5, j6)`
//------------------------------------------------------------------------------
//
// Three-term recurrence for function h(j) with `jmin == 0`.
//
template <typename T, typename ArgT>
struct funcs_xyz_wigner6j_rec_j_jmin0
{
    funcs_xyz_wigner6j_rec_j_jmin0(ArgT j2, ArgT j3, ArgT j4, ArgT j5, ArgT j6)
        : e2(castTo<T>(j2 + j3 + 1)),
          e4(castTo<T>(j5 + j6 + 1)),
          f2(castTo<T>(j2) * castTo<T>(j2 + 1)),
          f4(castTo<T>(j4) * castTo<T>(j4 + 1)),
          f5(castTo<T>(j5) * castTo<T>(j5 + 1))
    {
    }

    T X(T j) const
    {
        return Z(j + 1);
    }

    T Y(T j) const
    {
        // (2j+1)[-j(j+1) + j1(j1+1) + j2(j2+1) + l1(l1 + 1) + l2(l2+1)]
        return (2 * j + 1) * (-j * (j + 1) + 2 * (f2 - f4 + f5));
    }

    T Z(T j) const
    {
        return j * std::sqrt((j - e2) * (j + e2) * (j - e4) * (j + e4));
    }

private:
    T e2, e4;
    T f2, f4, f5;
};

//
// Three-term recurrence for function h(j) with `jmin > 0`.
//
template <typename T, typename ArgT>
struct funcs_xyz_wigner6j_rec_j
{
    funcs_xyz_wigner6j_rec_j(ArgT j2, ArgT j3, ArgT j4, ArgT j5, ArgT j6)
        : e1(castTo<T>(j2 - j3)),
          e2(castTo<T>(j2 + j3 + 1)),
          e3(castTo<T>(j5 - j6)),
          e4(castTo<T>(j5 + j6 + 1)),
          f2(castTo<T>(j2) * castTo<T>(j2 + 1)),
          f3(castTo<T>(j3) * castTo<T>(j3 + 1)),
          f4(castTo<T>(j4) * castTo<T>(j4 + 1)),
          f5(castTo<T>(j5) * castTo<T>(j5 + 1)),
          f6(castTo<T>(j6) * castTo<T>(j6 + 1))
    {
    }

    T aux(T j) const
    {
        return std::sqrt((j - e1) * (j + e1) * (j - e2) * (j + e2) * (j - e3) *
                         (j + e3) * (j - e4) * (j + e4));
    }

    T X(T j) const
    {
        return j * aux(j + 1);
    }

    T Y(T j) const
    {
        const auto x = j * (j + 1);
        return (2 * j + 1) * (x * (-x + f2 + f3 - 2 * f4) + f5 * (x + f2 - f3) +
                              f6 * (x - f2 + f3));
    };

    T Z(T j) const
    {
        return (j + 1) * aux(j);
    }

private:
    T e1, e2, e3, e4;
    T f2, f3, f4, f5, f6;
};
//
// implementation body
//
template <typename T, typename ArgT>
struct wigner6j_rec_j_impl
{
    using Scalar       = T;
    using ArgumentType = ArgT;
    using IntegerType  = typename IntegerTypeOf<ArgT>::type;

    static std::tuple<ArgumentType, ArgumentType>
    args_range(ArgumentType j2, ArgumentType j3, ArgumentType j4,
               ArgumentType j5, ArgumentType j6)
    {
        if (check_triad(j4, j2, j6) && check_triad(j4, j5, j3))
        {
            //
            // jmin/jmax are determined so that the triads (j1, j5, j) and
            // (j4, j2, j1) satisfy triangular condition.
            //
            const auto jmin = std::max(abs(j2 - j3), abs(j5 - j6));
            const auto jmax = std::min(j2 + j3, j5 + j6);
            return std::make_tuple(jmin, jmax);
        }
        // All symbols are zero for given arguments. Return negative value
        // pairs satisfying jmin = jmax + 1
        return std::make_tuple(ArgumentType(-1), ArgumentType(-2));
    }

    template <typename SizeT>
    static SizeT storage_size(ArgumentType jmin, ArgumentType jmax,
                              ArgumentType, ArgumentType, ArgumentType,
                              ArgumentType, ArgumentType)
    {
        return castTo<SizeT>(jmax - jmin + 1);
    }

    template <typename RandomAccessIterator>
    static void run(RandomAccessIterator out, ArgumentType jmin,
                    ArgumentType jmax, ArgumentType j2, ArgumentType j3,
                    ArgumentType j4, ArgumentType j5, ArgumentType j6)
    {
        using difference_type = typename std::iterator_traits<
            RandomAccessIterator>::difference_type;
        constexpr const Scalar sign[2] = {Scalar(1), Scalar(-1)};

        auto parity    = castTo<IntegerType>(j2 + j3 + j5 + j6);
        auto jj1       = castTo<Scalar>(twicePlusOne(jmin));
        const auto jj4 = castTo<T>(twicePlusOne(j4));
        if (jmin == jmax)
        {
            // Only one allowed symbol.
            *out = sign[parity & 1] / std::sqrt(jj1 * jj4);
            return;
        }

        if (jmin == 0)
        {
            //
            // This is reached when j2 = j3 and j5 = j6. In this case, j(j+1) is
            // factor out from X(j), Y(j) and Z(j): otherwise, X(jmin) becomes
            // zero and the recursion fails.
            //
            funcs_xyz_wigner6j_rec_j_jmin0<T, ArgT> xyz(j2, j3, j4, j5, j6);
            rec_luscombe_luban(jmin, jmax, out, xyz);
        }
        else
        {
            funcs_xyz_wigner6j_rec_j<T, ArgT> xyz(j2, j3, j4, j5, j6);
            rec_luscombe_luban(jmin, jmax, out, xyz);
        }
        //
        // Normalization
        //
        auto sum  = Scalar();
        auto last = out + castTo<difference_type>(jmax - jmin) + 1;
        for (auto it = out; it != last; ++it)
        {
            sum += jj1 * (*it) * (*it);
            jj1 += Scalar(2);
        }

        sum *= jj4;
        parity += static_cast<difference_type>(*(last - 1) < T());
        const auto scale = sign[parity & 1] / std::sqrt(sum);

        *out *= scale;
        while (++out != last)
        {
            *out *= scale;
        }
    }
};

} // namespace: detail

///
/// ## WignerSeriesContainer
///
/// This is a base class for holding the values of Wigner 3-*j* and 6-*j* series
/// computed by the recursion algorithms.
///
template <typename T, typename ArgT, typename ImplT>
class WignerSeriesContainer
{
public:
    using Scalar             = T;
    using ArgumentType       = ArgT;
    using IntegerType        = typename IntegerTypeOf<ArgT>::type;
    using ImplementationType = ImplT;
    using ContainerType      = std::vector<Scalar>;
    using Index              = typename ContainerType::size_type;

    explicit WignerSeriesContainer(Index capacity = 51)
        : data_(capacity), nmin_(), nmax_(-1)
    {
    }

    WignerSeriesContainer(const WignerSeriesContainer&) = default;
    WignerSeriesContainer(WignerSeriesContainer&&)      = default;
    ~WignerSeriesContainer()                            = default;
    WignerSeriesContainer& operator=(const WignerSeriesContainer&) = default;
    WignerSeriesContainer& operator=(WignerSeriesContainer&&) = default;

    ///
    /// Return the number of non-zero values
    ///
    Index size() const
    {
        return castTo<Index>(nmax_ - nmin_) + 1;
    }
    ///
    /// Check wherther the container is empty
    ///
    bool empty() const
    {
        return nmin_ > nmax_;
    }

    ///
    /// Return the lower bound of argument
    ///
    ArgumentType nmin() const
    {
        return nmin_;
    }
    ///
    /// Return the upper bound of argument
    ///
    ArgumentType nmax() const
    {
        return nmax_;
    }
    ///
    /// Return the value corresponding to given argument.  It returns the
    /// pre-computed value if `nmin() <= n && n <= nmax()`, otherwise zero.
    ///
    Scalar get(ArgumentType n) const
    {
        return (nmin() <= n && n <= nmax()) ? data_[get_index(n)] : Scalar();
    }
    ///
    /// Reference to the pre-computed values corresponding to argument `n`.
    /// `nmin() <= n && n <= nmax()` must be hold, otherwize the behavior is
    /// undefined.
    ///
    const Scalar& unsafeRef(ArgumentType n) const
    {
        assert(nmin() <= n && n <= nmax());
        return data_[get_index(n)];
    }
    ///
    /// Reserves storage
    ///
    void reserve(Index new_capacity)
    {
        data_.resize(new_capacity);
    }
    ///
    /// Compute series of function values
    ///
    template <typename... ArgsT>
    void compute(ArgsT... args)
    {
        std::tie(nmin_, nmax_) =
            ImplementationType::args_range(ArgumentType(args)...);

        if (!(nmin_ <= nmax_))
        {
            return;
        }

        const auto n = ImplementationType::template storage_size<Index>(
            nmin_, nmax_, ArgumentType(args)...);

        expand_if_necessary(n);

        ImplementationType::run(std::begin(data_), nmin_, nmax_,
                                ArgumentType(args)...);
    }
    ///
    /// Requests the removal of unused capacity.
    ///
    void shrink_to_fit()
    {
        if (data_.size() != size())
        {
            ContainerType tmp(std::begin(data_), std::begin(data_) + size());
            data_ = std::move(tmp);
        }
    }

protected:
    Index get_index(ArgumentType n) const
    {
        return castTo<Index>(n - nmin());
    }

    void expand_if_necessary(Index n)
    {
        if (data_.size() < n)
        {
            data_.resize(n);
        }
    }

    ContainerType data_;
    ArgumentType nmin_;
    ArgumentType nmax_;
};

///
/// ## Wigner3jSeriesJ
///
/// Compute Wigner 3-*j* symbols over $$j_{3}$$
///
/// ### Description
///
/// For given arguments $$j_1, j_2, m_1, m_2$$, this class compute the Wigner
/// 3-*j* symbols of the form,
///
/// ```math
///    f(j) = \left( \begin{array}{ccc}
///               j_1 & j_2 & j \\
///               m_1 & m_2 & -m_1-m_2
///           \end{array}{ccc} \right)
/// ```
///
/// for all possible $$j$$ such that $$f(j)$$ becomes non-zero. The lower and
/// upper bound of $$j$$ are given as
///
/// ```math
///   j_{\mathrm{min}} &= \max(|j_1 - j_2|, |m_1 + m_2|) \\
///   j_{\mathrm{max}} &= j_1 + j_2.
/// ```
///
/// $$f(j)$$ values are calculated using the three-term recurrence relations
/// derived by Schulten and Gordon, which is given as
///
/// ```math
///   j A(j+1) f(j+1) + B(j) f(j) + (j+1) A(j) f(j-1) = 0, \\
/// ```
///
/// where,
///
/// ```math
///   A(j) &= \sqrt{[j^2-(j_1-j_2)^2][(j_1+j_2+1)^2-j^2][j^2-(m_1+m_2)^2]}, \\
///   B(j) &= (2j+1)[(m_1+m_2)\{j_1(j_1+1)-j_2(j_2+1)\}-(m_1-m_2)j(j+1)].
/// ```
///
/// The values of $$f(j)$$ are normalized with the phase convension, which are
/// described as
///
/// ```math
///  \sum_{j=j_{\mathrm{min}}}^{j_{\mathrm{max}}}(2j+1)f^{2}(j)=1, \\
///  \sgn[f(j_{\mathrm{max}})]=(-1)^{j_1-j_2+m_1+m_2}.
/// ```
///
template <typename T, typename ArgT>
using Wigner3jSeriesJ =
    WignerSeriesContainer<T, ArgT, detail::wigner3j_rec_j_impl<T, ArgT>>;

///
/// ## Wigner3jSeriesM
///
/// Compute Wigner 3-*j* symbols over $$m_{2}$$
///
/// ### Description
///
/// For given arguments $$j_1, j_2, j_1, m_1$$, this class compute the Wigner
/// 3-*j* symbols of the form,
///
/// ```math
///    g(m) = \left( \begin{array}{ccc}
///               j_1 & j_2 & j_3 \\
///               m_1 & m   & -m-m_1
///           \end{array}{ccc} \right)
/// ```
///
/// for all possible $$m$$ such that $$g(m)$$ becomes non-zero. The lower and
/// upper bound of $$j$$ are given as
///
/// ```math
///   m_{\mathrm{min}} &= \max(-j_2, -j_3 - m_1|) \\
///   m_{\mathrm{max}} &= \min(j_2, j_3 - m_1)
/// ```
///
/// $$g(m)$$ values are calculated using the three-term recurrence relations
/// derived by Schulten and Gordon, which is given as
///
/// ```math
///   C(m+1) g(m+1) + D(m) g(m) + C(m) g(m-1) = 0, \\
/// ```
///
/// where,
///
/// ```math
///   C(m) &= \sqrt{(j_2 - m + 1)(j_2 + m)(j_3 - m - m_1 + 1)(j_3 + m + m_1)}
///   D(m) &= j_2 (j_2 + 1) + j_3 (j_3 + 1) - j_1 (j_1 + 1) - 2m (m + m_1)
/// ```
///
/// The values of $$g(m)$$ are normalized with the phase convension, which are
/// described as
///
/// ```math
///  (2j_1 + 1)\sum_{m=m_{\mathrm{min}}}^{m_{\mathrm{max}}})g^{2}(m)=1, \\
///  \sgn[g(m_{\mathrm{max}})] = (-1)^{j_2 - j_3 - m_1}.
/// ```
///
template <typename T, typename ArgT>
using Wigner3jSeriesM =
    WignerSeriesContainer<T, ArgT, detail::wigner3j_rec_m_impl<T, ArgT>>;

///
/// ## Wigner6jSeriesJ
///
/// Compute Wigner 6-*j* symbols over $$j_{1}$$
///
/// ### Description
///
/// For given arguments $$j_2, j_3, j_4, j_5, j_6$$, this class compute the
/// Wigner 6-*j* symbols of the form,
///
/// ```math
///    h(j) = \left\{ \begin{array}{ccc}
///               j   & j_2 & j_3 \\
///               j_4 & j_5 & j_6
///           \end{array}{ccc} \right\}
/// ```
///
/// for all possible $$j$$ such that $$h(j)$$ becomes non-zero. The lower and
/// upper bound of $$j$$ are given as
///
/// ```math
///   j_{\mathrm{min}} &= \max(|j_2 - j_3|, |j5 - j_6|) \\
///   j_{\mathrm{max}} &= \min(j_2 + j_3, j5 + j_6)
/// ```
///
/// $$h(j)$$ values are calculated using the three-term recurrence relations
/// derived by Schulten and Gordon, which is given as
///
/// ```math
///   j E(j+1) h(j+1) + F(j) h(j) + (j+1) E(j) h(j-1) = 0, \\
/// ```
///
/// where,
///
/// ```math
///   E(j) = &\sqrt{[j^2 - (j_2 - j_3)^2] [(j_2 + j_3 + 1)^2 - j^2]
///                 [j^2 - (j_5 - j_6)^2] [(j_5 + j_6 + 1)^2 - j^2]}, \\
///   F(j) = & (2j + 1) \{ t(j)   [-t(j) + t(j_2) + t(j_3) - 2 t(j_4)]
///                      + t(j_5) [ t(j) + t(j_2) - t(j_3)]
///                      + t(j_6) [ t(j) - t(j_2) + t(j_3)] \}, \\
///   t(j) = & j (j + 1).
/// ```
///
/// The values of $$h(j)$$ are normalized with the phase convension, which are
/// described as
///
/// ```math
///  (2 j_4 + 1)\sum_{j=j_{\mathrm{min}}}^{j_{\mathrm{max}}}(2j+1)h^{2}(j)=1, \\
///  \sgn[h(j_{\mathrm{max}})]=(-1)^{j_2 + j_3 + j_5 + j_6}.
/// ```
///
template <typename T, typename ArgT>
using Wigner6jSeriesJ =
    WignerSeriesContainer<T, ArgT, detail::wigner6j_rec_j_impl<T, ArgT>>;

///
/// ## Wigner9j
///
/// Wigner 9-j symbol calculatlor.
///
/// ### Description
///
/// This class calculate Wigner 9-*j* symbols using the expansion in terms of
/// *6*-j symbols, which is given as
///
/// ```math
///   \left\{ \begin{array}{ccc}
///     j_1 & j_2 & j_3 \\
///     j_4 & j_5 & j_6 \\
///     j_7 & j_8 & j_9 \\
///     \end{array} \right\}
///   &= \sum_{x}(-1)^{2x}(2x+1)
///     \left\{ \begin{array}{ccc}
///       j_1 & j_4 & j_7 \\
///       j_8 & j_9 & x   \\
///     \end{array} \right\}
///     \left\{ \begin{array}{ccc}
///       j_2 & j_5 & j_8 \\
///       j_4 &  x  & j_6 \\
///     \end{array} \right\}
///     \left\{ \begin{array}{ccc}
///       j_3 & j_6 & j_9 \\
///       x &  j_1  & j_2 \\
///     \end{array} \right\}.
/// ```
///
template <typename T, typename ArgT>
class Wigner9j
{
public:
    using Scalar       = T;
    using ArgumentType = ArgT;
    using IntegerType  = typename IntegerTypeOf<ArgT>::type;

private:
    using Container6j = Wigner6jSeriesJ<T, ArgT>;

public:
    using Index = typename Container6j::Index;

    explicit Wigner9j(Index capacity = 51)
        : w6j_{Container6j(capacity), Container6j(capacity),
               Container6j(capacity)}
    {
    }

    Wigner9j(const Wigner9j&) = default;
    Wigner9j(Wigner9j&&)      = default;
    ~Wigner9j()               = default;

    Wigner9j& operator=(const Wigner9j&) = default;
    Wigner9j& operator=(Wigner9j&&) = default;

    Scalar compute(ArgumentType j1, ArgumentType j2, ArgumentType j3,
                   ArgumentType j4, ArgumentType j5, ArgumentType j6,
                   ArgumentType j7, ArgumentType j8, ArgumentType j9);

    template <typename Arg1, typename Arg2, typename Arg3, //
              typename Arg4, typename Arg5, typename Arg6, //
              typename Arg7, typename Arg8, typename Arg9>
    typename std::enable_if<(std::is_convertible<Arg1, ArgumentType>::value &&
                             std::is_convertible<Arg2, ArgumentType>::value &&
                             std::is_convertible<Arg3, ArgumentType>::value &&
                             std::is_convertible<Arg4, ArgumentType>::value &&
                             std::is_convertible<Arg5, ArgumentType>::value &&
                             std::is_convertible<Arg6, ArgumentType>::value &&
                             std::is_convertible<Arg7, ArgumentType>::value &&
                             std::is_convertible<Arg8, ArgumentType>::value &&
                             std::is_convertible<Arg9, ArgumentType>::value),
                            Scalar>::type
    compute(Arg1 j1, Arg2 j2, Arg3 j3, //
            Arg4 j4, Arg5 j5, Arg6 j6, //
            Arg7 j7, Arg8 j8, Arg9 j9)
    {
        return eval(ArgumentType(j1), ArgumentType(j2), ArgumentType(j3),
                    ArgumentType(j4), ArgumentType(j5), ArgumentType(j6),
                    ArgumentType(j7), ArgumentType(j8), ArgumentType(j9));
    }

private:
    constexpr static Scalar sign_for_sum(IntegerType)
    {
        return Scalar(1);
    }

    static Scalar sign_for_sum(HalfInteger<IntegerType> x)
    {
        return x.isHalfInteger() ? Scalar(-1) : Scalar(1);
    }

    Container6j w6j_[3];
};

template <typename T, typename ArgT>
T Wigner9j<T, ArgT>::compute(ArgumentType j1, ArgumentType j2, ArgumentType j3,
                             ArgumentType j4, ArgumentType j5, ArgumentType j6,
                             ArgumentType j7, ArgumentType j8, ArgumentType j9)
{
    w6j_[0].compute(j1, j9, j8, j4, j7);
    if (w6j_[0].empty())
    {
        return Scalar(/*zero*/);
    }

    w6j_[1].compute(j2, j6, j5, j4, j8);
    if (w6j_[1].empty())
    {
        return Scalar(/*zero*/);
    }

    w6j_[2].compute(j1, j9, j3, j6, j2);
    if (w6j_[2].empty())
    {
        return Scalar(/*zero*/);
    }

    auto x    = std::max({w6j_[0].nmin(), w6j_[1].nmin(), w6j_[2].nmin()});
    auto xmax = std::min({w6j_[0].nmax(), w6j_[1].nmax(), w6j_[2].nmax()});
    const auto sign = sign_for_sum(x);

    auto result = T();
    auto weight = castTo<T>(twicePlusOne(x));

    while (x <= xmax)
    {
        result += weight * w6j_[0].unsafeRef(x) * w6j_[1].unsafeRef(x) *
                  w6j_[2].unsafeRef(x);
        weight += Scalar(2);
        ++x;
    }

    return sign * result;
}

} // namespace: wigner

#endif /* WIGNER_WIGNER_THREE_NJ_HPP */
