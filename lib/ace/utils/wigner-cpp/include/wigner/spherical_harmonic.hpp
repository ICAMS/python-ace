// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-

/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Hidekazu Ikeno
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * * of this software and associated documentation files (the "Software"), to
 * deal * in the Software without restriction, including without limitation the
 * rights
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
/// \file wigner/spherical_harmonics.hpp
///
/// Implement spherical harmonics calculator
///

#ifndef WIGNER_SPHERICAL_HARMONICS_HPP
#define WIGNER_SPHERICAL_HARMONICS_HPP

// -- Header files of standard libraries
#include <cassert>
#include <stdexcept>
#include <vector>

#include "wigner/constants.hpp"
#include "wigner/misc.hpp"

namespace wigner
{

///
/// Helper functions for manipulating arrays storing normalized associated
/// Legendre polynomials, $$\overline{P}_{lm} \, (0 \leq m \leq l).$$
///
struct LegendreArrayHelper
{
    ///
    /// Check whether the given pair of degree `l` and order `m` satisfy
    /// $$l \geq 0$$ and $$ 0 \leq m \leq l. $$
    ///
    /// @param l degree of associate Legendre polynomial
    /// @param m order of associate Legendre polynomial
    ///
    static constexpr bool is_valid_args(int l, int m)
    {
        return 0 <= l && 0 <= m && m <= l;
    }
    ///
    /// Return the number of associated Legendre polynomials $$P_{lm}$$ non-zero
    /// upto degree `l_max`.
    ///
    /// @param l_max  maximum degree of associate Legendre polynomial to be
    ///               computed. `l_max >= 0` required.
    ///
    static constexpr std::size_t array_size(int l_max)
    {
        return static_cast<std::size_t>((l_max + 1) * (l_max + 2) / 2);
    }
    ///
    /// Return the index for the (l,m) compoenent of normalized associate
    /// Legendre polynomial. `is_valid_args(l, m)` should be `true`, otherwise
    /// the returned index may be out-of-range.
    ///
    /// @param l degree of associate Legendre polynomial
    /// @param m order of associate Legendre polynomial
    ///
    static constexpr std::size_t index(int l, int m)
    {
        return static_cast<std::size_t>(l * (l + 1) / 2 + m);
    }
};

///
/// Helper functions for manipulating arrays storing spherical harmonics,
/// $$Y_{lm} \, (l \geq 0, \, -l \leq m \leq l).$$
///
struct SphericalHarmonicArrayHelper
{
    ///
    /// Check whether the given pair of degree `l` and order `m` satisfy
    /// $$l \geq 0$$ and $$ -l \leq m \leq l. $$
    ///
    /// @param l degree of spherical harmonics
    /// @param m order of spherical harmonics
    ///
    static constexpr bool is_valid_args(int l, int m)
    {
        return 0 <= l && std::abs(m) <= l;
    }

    ///
    /// Return the number of spherical harmonics $$Y_{lm}$$ non-zero upto degree
    /// `l_max`.
    ///
    /// @param l_max  maximum degree of spherical harmonics to be
    ///               computed. `l_max >= 0` required.
    ///
    static constexpr std::size_t array_size(int l_max)
    {
        return static_cast<std::size_t>((l_max + 1) * (l_max + 1));
    }

    ///
    /// Return the index for the (l,m) compoenent of normalized associate
    /// Legendre polynomial. `is_valid_args(l, m)` should be `true`, otherwise
    /// the returned index may be out-of-range.
    ///
    /// @param l  degree of spherical harmonics
    /// @param m  order of spherical harmonics
    ///
    static constexpr std::size_t index(int l, int m)
    {
        return static_cast<std::size_t>(l * (l + 1) + m);
    }
};

///
/// Basic container class for storing spherical functions such as associated
/// Legendre polynomials and spherial harmonics for all (l,m) pairs.
///
template <typename T, typename HelperT>
class SphericalFunctionArrayBase
{
public:
    using Scalar        = T;
    using RealScalar    = typename RealTypeOf<T>::type;
    using ContainerType = std::vector<Scalar>;
    using Index         = typename ContainerType::size_type;
    using Pointer       = typename ContainerType::pointer;
    using ConstPointer  = typename ContainerType::const_pointer;
    using ArrayHelper   = HelperT;

    SphericalFunctionArrayBase() = default;

    explicit SphericalFunctionArrayBase(int l_max)
        : lmax_(l_max), data_(ArrayHelper::array_size(l_max))
    {
    }

    SphericalFunctionArrayBase(const SphericalFunctionArrayBase&) = default;

    SphericalFunctionArrayBase(SphericalFunctionArrayBase&&) = default;

    ~SphericalFunctionArrayBase() = default;

    SphericalFunctionArrayBase&
    operator=(const SphericalFunctionArrayBase&) = default;

    SphericalFunctionArrayBase&
    operator=(SphericalFunctionArrayBase&&) = default;

    int maxDegree() const
    {
        return lmax_;
    }

    Scalar get(int l, int m) const
    {
        return ArrayHelper::is_valid_args(l, m) && l <= maxDegree()
                   ? unsafeRef(l, m)
                   : Scalar();
    }

    bool set(int l, int m, const Scalar& val)
    {
        if (ArrayHelper::is_valid_args(l, m) && l <= maxDegree())
        {
            unsafeRef(l, m) = val;
            return true;
        }
        else
        {
            return false;
        }
    }

    Scalar& unsafeRef(int l, int m)
    {
        return data_[ArrayHelper::index(l, m)];
    }

    const Scalar& unsafeRef(int l, int m) const
    {
        return data_[ArrayHelper::index(l, m)];
    }

    void resize(int l_max)
    {
        const auto n = ArrayHelper::array_size(l_max);
        data_.resize(n);
        lmax_ = l_max;
    }

    Pointer data()
    {
        return data_.data();
    }

    ConstPointer data() const
    {
        return data_.data();
    }

private:
    int lmax_;
    ContainerType data_;
};

///
/// ContainerType class for storing normalized associated Legendre functions
/// $$\overline{P}_{l}^{m}(x)$$ for $$0 \leq l \leq l_{\text{max}}$$ and
/// $$0 \leq m \leq l$$
///
template <typename T>
using LegendreArray = SphericalFunctionArrayBase<T, LegendreArrayHelper>;

///
/// ContainerType class for storing spherical harmonics
/// $$\overline{Y}_{l}^{m}(x)$$ for $$0 \leq l \leq l_{\text{max}}$$ and
/// $$-l \leq m \leq l$$
///
template <typename T>
using SphericalHarmonicArray =
    SphericalFunctionArrayBase<T, SphericalHarmonicArrayHelper>;

//==============================================================================
// Normalized associated Legendre polynomial
//==============================================================================
namespace detail
{

//
// Scaling factor normalized associated Legendre polynomials, which is applied
// so as to prevent from overflow/underflows of floating point values for large
// degree.
//
template <typename T>
constexpr T legendre_global_scaling()
{
    constexpr T scale = T(1) / T(std::numeric_limits<T>::radix);
    int count         = (std::numeric_limits<T>::max_exponent * 7) / 8;
    auto x            = scale;
    while (--count)
    {
        x *= scale;
    }
    return x;
}

} // namespace: detail

///
/// Enum for dispatching normalization convention of spherical hamonics.
///
enum NormalizationConvention : std::uint32_t
{
    CONDON_SHORTLEY_PHASE       = 1u,
    ORTHONORMALIZED             = 1u << 1,
    FOUR_PI_NORMALIZED          = 1u << 2,
    SCHMIDT_SEMI_NORMALIZED     = 1u << 3,
    NORMALIZED_LEGENDRE_DEFAULT = FOUR_PI_NORMALIZED | CONDON_SHORTLEY_PHASE,
    SPHERICAL_HARMONIC_DEFAULT  = ORTHONORMALIZED | CONDON_SHORTLEY_PHASE,
};

///
/// Compute normalized associated Legendre polynomials
///
template <typename T, std::uint32_t Convention = NORMALIZED_LEGENDRE_DEFAULT,
          typename HelperT = LegendreArrayHelper>
class NormalizedLegendreCalculator
{
public:
    using Scalar      = T;
    using RealScalar  = typename RealTypeOf<T>::type;
    using Index       = std::size_t;
    using ArrayHelper = HelperT;

private:
    // Pre-factor for real spherical harmonics
    constexpr static auto factor_for_real_harmonics =
        IsComplex<T>::value ? RealScalar(1) : math_const<RealScalar>::sqrt2;
    // Constant in normalization factor
    constexpr static auto norm_const =
        (Convention & ORTHONORMALIZED)
            ? RealScalar(0.5) /
                  math_const<RealScalar>::sqrt_pi // 1 / sqrt(4\pi)
            : RealScalar(1);
    // P_{1}^{1}(x) / u :  value of (1,1) component
    constexpr static auto h11 =
        (Convention & SCHMIDT_SEMI_NORMALIZED)
            ? RealScalar(1) / math_const<RealScalar>::sqrt2
            : math_const<RealScalar>::sqrt3 / math_const<RealScalar>::sqrt2;
    // Condon-Shortley phase
    constexpr static auto cs_phase =
        (Convention & CONDON_SHORTLEY_PHASE) ? RealScalar(-1) : RealScalar(1);
    // Global scaling factor to prevent overflow/underflow
    constexpr static auto g_scale =
        detail::legendre_global_scaling<RealScalar>();

public:
    NormalizedLegendreCalculator() = default;

    explicit NormalizedLegendreCalculator(int l_max)
        : lmax_(l_max), isqrt_(static_cast<Index>(2 * l_max + 2))
    {
        set_table();
    }

    NormalizedLegendreCalculator(const NormalizedLegendreCalculator&) = default;
    NormalizedLegendreCalculator(NormalizedLegendreCalculator&&)      = default;
    ~NormalizedLegendreCalculator()                                   = default;
    NormalizedLegendreCalculator&
    operator=(const NormalizedLegendreCalculator&) = default;
    NormalizedLegendreCalculator&
    operator=(NormalizedLegendreCalculator&&) = default;
    ///
    /// Compute the index of 1D array corresponding to `(l,m)` component,
    /// $$Y_{l}^{m}.$$
    ///
    constexpr static Index index(int l, int m)
    {
        return ArrayHelper::index(l, m);
    }
    ///
    /// Maximum degree of spherical harmonics, $$l_{\text{max}},$$ to be
    /// computed.
    ///
    int maxDegree() const
    {
        return lmax_;
    }
    ///
    /// Get the number of associated Legendre polynomials up to `maxDegree().`
    ///
    Index arraySize() const
    {
        return ArrayHelper::array_size(maxDegree());
    }
    ///
    /// Compute associated Legendre polynomials $$Y_{l}^{m}(x)$$ at a
    /// given point `x` for `0 <= l && l <= maxDegree()` and
    /// `0 <= m && m <= l.`
    ///
    /// @param x    argument in $$[-1,1]$$
    /// @param out  the beginning of the destination that store $$P_{l}^{m}$$
    ///             values
    ///
    template <typename RandomAccessIterator>
    void compute(RealScalar x, RandomAccessIterator out) const;
    ///
    /// Compute associated Legendre polynomials $$P_{l}^{m}(x)$$ at a given
    /// point and store them into a `SphericalFunctionArrayBase`
    ///
    void compute(RealScalar x,
                 SphericalFunctionArrayBase<T, ArrayHelper>& out) const
    {
        assert(maxDegree() <= out.maxDegree());
        compute(x, out.data());
    }
    ///
    /// Set maximum degree of spherical harmonics to be computed.
    ///
    void setMaxDegree(int l_max)
    {
        lmax_ = l_max;
        isqrt_.resize(static_cast<Index>(2 * l_max + 2));
        set_table();
    }

private:
    void set_table()
    {
        auto k = 0;
        for (auto& x : isqrt_)
        {
            x = std::sqrt(static_cast<RealScalar>(k));
            ++k;
        }
    }

    RealScalar sectoral_next(int l, RealScalar prev,
                             std::integral_constant<std::uint32_t, 0u>) const
    {
        return isqrt_[static_cast<Index>(2 * l + 1)] /
               isqrt_[static_cast<Index>(2 * l)] * prev;
    }

    RealScalar sectoral_next(
        int l, RealScalar prev,
        std::integral_constant<std::uint32_t, SCHMIDT_SEMI_NORMALIZED>) const
    {
        return isqrt_[static_cast<Index>(2 * l - 1)] /
               isqrt_[static_cast<Index>(2 * l)] * prev;
    }

    int lmax_ = -1;
    std::vector<RealScalar> isqrt_;
};

template <typename T, std::uint32_t Convention, typename ArrayHelper>
template <typename RandomAccessIterator>
void NormalizedLegendreCalculator<T, Convention, ArrayHelper>::compute(
    RealScalar x, RandomAccessIterator out) const
{
    using normalization_dispatch =
        std::integral_constant<std::uint32_t,
                               Convention & SCHMIDT_SEMI_NORMALIZED>;

    const auto uu = (1 - x) * (1 + x);
    const auto u  = std::sqrt(uu);

    // l = 0
    auto hll = norm_const; // H_{ll} = P_{ll}(t)/u^{l}
    *out     = hll;
    if (maxDegree() == 0)
    {
        return;
    }

    // l = 1
    hll       = cs_phase * norm_const * factor_for_real_harmonics * h11;
    auto dest = out + index(1, 1);
    *dest     = hll * u;
    *(--dest) = cs_phase * math_const<RealScalar>::sqrt2 /
                factor_for_real_harmonics * x * hll;
    if (maxDegree() == 1)
    {
        return;
    }

    // Values of P_{l}^{m}(t), P_{l}^{m+1}(t), P_{l}^{m+2}(t), respectively
    RealScalar plm, plm_pre1, plm_pre2;
    hll *= g_scale;
    const auto isqrt = std::begin(isqrt_);

    for (int l = 2; l <= maxDegree(); ++l)
    {
        // m = l: hll = g_scale * P_{ll}(t) / u^{l}
        hll      = cs_phase * sectoral_next(l, hll, normalization_dispatch());
        dest     = out + index(l, l);
        *dest    = hll;
        plm_pre1 = hll;

        // m = l - 1
        int m     = l - 1;
        plm       = cs_phase * isqrt[2 * l] * x * plm_pre1;
        *(--dest) = plm;

        // m = l-2,...,1
        while (--m > 0)
        {
            const auto cm = RealScalar(1) / (isqrt[l - m] * isqrt[l + m + 1]);
            const auto am = cs_phase * 2 * (m + 1) * cm;
            const auto bm = -isqrt[l + m + 2] * isqrt[l - m - 1] * cm;
            plm_pre2 = plm_pre1; // plm_pre2 := g_scale * P_{l,m+2}(t) / u^{m+2}
            plm_pre1 = plm;      // plm_pre1 := g_scale * P_{l,m+1}(t) / u^{m+1}
            plm      = am * x * plm_pre1 + bm * uu * plm_pre2;
            *(--dest) = plm;
        }

        // m = 0
        const auto c0 = RealScalar(1) /
                        (factor_for_real_harmonics * isqrt[l] * isqrt[l + 1]);
        const auto a0 = cs_phase * 2 * c0;
        const auto b0 = -isqrt[l + 2] * isqrt[l - 1] * c0;
        plm_pre2      = plm_pre1;
        plm_pre1      = plm;
        plm           = a0 * x * plm_pre1 + b0 * uu * plm_pre2;
        *(--dest)     = plm;

        auto rescale = 1 / g_scale; // rescale = u^{m} / g_scale
        *dest *= rescale;           // for m = 0
        while (++m <= l)
        {
            rescale *= u;
            *(++dest) *= rescale;
        }
    }
}

namespace detail
{
//
// Trigonmetric part for spherical harmonics $$\Phi_{m}(\phi)$$
//
// When `T` is real number type,
//
// ```math
//  \Phi_{m}(\phi) =
//  \begin{cases}
//     \cos|m|\phi & (m > 0), \\
//     1           & (m = 0), \\
//     \sin|m|\phi & (m < 0).
//  \end{cases}
// ```
//
// When `T` is a complex number type,
//
// ```math
//  \Phi_{m}(\phi) =
//  \begin{cases}
//     \exp(im\phi)        & (m > 0), \\
//     1                   & (m = 0), \\
//     (-1)^{m}\exp(imphi) & (m < 0).
//  \end{cases}
// ```
//
// The Condon-Shortley phase factor $$(-1)^{m}$$ is omitted if
// `condon_shortley_phase` bit is not set on the template argument value `C`.
//
template <typename T, std::uint32_t C>
struct sph_harm_trigonometric_part
{
    template <typename Iterator>
    static void compute(int lmax, T phi, Iterator out)
    {
        assert(lmax >= 0);
        auto cm = out + lmax;
        *cm     = T(1);
        auto sn = cm;
        for (int m = 1; m <= lmax; ++m)
        {
            *(++cm) = std::cos(m * phi);
            *(--sn) = std::sin(m * phi);
        }
    }
};

template <typename T, std::uint32_t C>
struct sph_harm_trigonometric_part<std::complex<T>, C>
{
    constexpr static T phase = (C & CONDON_SHORTLEY_PHASE) ? T(-1) : T(1);
    template <typename Iterator>
    static void compute(int lmax, T phi, Iterator out)
    {
        constexpr static T sign[2] = {T(1), phase};
        assert(lmax >= 0);
        auto cm = out;
        std::advance(cm, lmax);
        *cm     = T(1);
        auto sm = cm;
        for (int m = 1; m <= lmax; ++m)
        {
            const auto cos_phi = std::cos(m * phi);
            const auto sin_phi = std::sin(m * phi);
            *(++cm)            = std::complex<T>(cos_phi, sin_phi);
            *(--sm) = sign[m & 1] * std::complex<T>(cos_phi, -sin_phi);
        }
    }
};

} // namespace: detail

///
/// Compute a set of spherical harmonics.
///
template <typename T, std::uint32_t Convention = SPHERICAL_HARMONIC_DEFAULT>
class SphericalHarmonicCalculator
{
public:
    using Scalar      = T;
    using RealScalar  = typename RealTypeOf<T>::type;
    using Index       = std::size_t;
    using ArrayHelper = SphericalHarmonicArrayHelper;

private:
    using legnedre_calculator =
        NormalizedLegendreCalculator<T, Convention, ArrayHelper>;

public:
    SphericalHarmonicCalculator() = default;

    explicit SphericalHarmonicCalculator(int l_max)
        : legendre_(l_max), f_m_phi_(static_cast<Index>(2 * l_max + 1))
    {
    }

    SphericalHarmonicCalculator(const SphericalHarmonicCalculator&) = default;
    SphericalHarmonicCalculator(SphericalHarmonicCalculator&&)      = default;
    SphericalHarmonicCalculator&
    operator=(const SphericalHarmonicCalculator&) = default;
    SphericalHarmonicCalculator&
    operator=(SphericalHarmonicCalculator&&) = default;
    ///
    /// Compute the index of 1D array corresponding to `(l,m)` component,
    /// $$Y_{l}^{m}.$$
    ///
    constexpr static Index index(int l, int m)
    {
        return ArrayHelper::index(l, m);
    }
    ///
    /// Maximum degree of spherical harmonics, $$l_{\text{max}},$$ to be
    /// computed.
    ///
    int maxDegree() const
    {
        return legendre_.maxDegree();
    }
    ///
    /// Get the number of spherical harmonics $$Y_{l}^{m}$$ for
    /// `0 <= l && l <= maxDegree()` and `abs(m) <= l.`
    ///
    Index arraySize() const
    {
        return ArrayHelper::array_size(maxDegree());
    }
    ///
    /// Compute spherical harmonics $$Y_{l}^{m}(\theta,\phi)$$ at a given point
    /// `(theta,phi)` for `0 <= l && l <= maxDegree()` and `abs(m) <= l.`
    ///
    /// @param theta  polar angle (colatitudinal coordinate) in $$[0,\pi]$$
    /// @param phi    azimuthal angle (longitudinal coordinate) in $$[0,2\pi)$$
    /// @param out    the beginning of the destination that store $$Y_{l}^{m}$$
    ///               values
    ///
    template <typename RandomAccessIterator>
    void compute(RealScalar theta, RealScalar phi, RandomAccessIterator out);
    ///
    /// Compute spherical harmonics $$Y_{l}^{m}(\theta,\phi)$$ at a given point
    /// and store them into a `SphericalHarmonicArray`
    ///
    void compute(RealScalar theta, RealScalar phi,
                 SphericalHarmonicArray<T>& out)
    {
        compute(theta, phi, out.data());
    }
    ///
    /// Set maximum degree of spherical harmonics to be computed.
    ///
    void setMaxDegree(int l_max)
    {
        legendre_.setMaxDegree(l_max);
        f_m_phi_.resize(static_cast<Index>(2 * l_max + 1));
    }

private:
    legnedre_calculator legendre_;
    std::vector<T> f_m_phi_; // (m,phi)-dependent part
};

template <typename T, std::uint32_t Convention>
template <typename RandomAccessIterator>
void SphericalHarmonicCalculator<T, Convention>::compute(
    RealScalar theta, RealScalar phi, RandomAccessIterator out)
{
    using trigonometric_part =
        detail::sph_harm_trigonometric_part<T, Convention>;
    // Compute normalized associated Legendre polynomials
    const auto cos_theta = std::cos(theta);
    legendre_.compute(cos_theta, out);
    // Compute trigonometric part depending on m and phi
    trigonometric_part::compute(maxDegree(), phi, std::begin(f_m_phi_));
    // Multiply both factors
    const auto f_m_phi = std::begin(f_m_phi_) + maxDegree();
    for (int l = 1; l <= maxDegree(); ++l)
    {
        auto dest = out + index(l, 0);
        for (int m = 1; m <= l; ++m)
        {
            const auto plm = dest[m];
            dest[m]        = plm * f_m_phi[m];
            dest[-m]       = plm * f_m_phi[-m];
        }
    }
}

} // namespace: wigner

#endif /* WIGNER_SPHERICAL_HARMONICS_HPP */
