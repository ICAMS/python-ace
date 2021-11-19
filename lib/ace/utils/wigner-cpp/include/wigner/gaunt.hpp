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
/// \file   wigner/gaunt.hpp
/// \brief  Gaunt coefficients calculator
///

#ifndef WIGNER_GAUNT_HPP
#define WIGNER_GAUNT_HPP

#include <cassert>
#include <cmath>
#include <cstdlib>

#include <limits>
#include <vector>

#include "wigner/constants.hpp"
#include "wigner/wigner_3nj.hpp"

namespace wigner
{

namespace detail
{

//
// ## gaunt_pre
//
// Pre-factor of Gaunt coefficients
//
// ### Description
//
// This function compute
//
//   f(j) = sqrt((2*j1+1) * (2*j2+1) * (2*j+1) / 4pi) * wigner3j(j1,j2,j,0,0,0)
//
// for all allowed range of j values,  jmin <= j <= jmax, where
//
//   jmin = |j1-j2|, jmax = j1 + j2, and j1 + j2 + j even.
//
template <typename IntT, typename RandomAccessIterator>
void gaunt_pre(IntT j1, IntT j2, RandomAccessIterator ret)
{
    using T = typename std::decay<decltype(*ret)>::type;

    constexpr static T pi4 = 4 * math_const<T>::pi;

    assert(j1 >= 0 && j2 >= 0);

    auto jmin = abs(j1 - j2);
    auto jmax = j1 + j2;
    auto pre  = std::sqrt(T(twicePlusOne(j1) * twicePlusOne(j2)) / pi4);

    if (jmin == jmax)
    {
        ret[0] = (jmin & IntT(1)) ? -pre : pre;
        return;
    }

    auto siz = (jmax - jmin) / 2 + 1;

    auto Ksq = [=](IntT j) {
        return castTo<T>((j - jmin) * (j + jmin) * (jmax + 1 - j) *
                         (jmax + 1 + j));
    };

    //-------------------------------------------------------------------------
    //
    // Apply the recurrence relation,
    //
    //   f(j-2) = [-K(j)/K(j-1)] f(j),
    //   K(j)   = sqrt{(j^2 - (j1-j2)^2) ((j1+j2+1)^2 - j^2)}
    //
    // from the both side, i.e., the recursion starts from j = jmin or jmax
    // to jc, where |f(j)| takes an extremum value.
    //
    //-------------------------------------------------------------------------
    //
    // ===== Forward recursion =====
    //
    auto j           = jmin;
    decltype(siz) kc = 0;
    while (j > jmin)
    {
        auto r = std::sqrt(Ksq(j + 2) / Ksq(j + 1));
        if (r < T(1))
            break;
        ret[kc] = -r;
        ++kc;
        j += 2;
    }
    //
    // Compute f(j) for jmin <= j <= jc apart from the normalization factor.
    //
    ret[kc]  = T(1);         // set f(jc) = 1
    auto sum = T(2 * j + 1); // normalization constant
    auto k   = kc - 1;
    while (j > jmin)
    {
        j -= 2;
        ret[k] *= ret[k + 1];
        sum += castTo<T>(twicePlusOne(j)) * ret[k] * ret[k];
        --k;
    }
    //
    // ===== Backward recursion =====
    //
    j = jmax;
    k = siz;

    while (--k > kc)
    {
        ret[k] = -std::sqrt(Ksq(j - 1) / Ksq(j));
        j -= 2;
    }
    //
    // Compute f(j) for jc <= j <= jmax apart from the normalization factor.
    //
    // Note that ret[kc] = f(jc) have already set to unity.
    //
    ++k;
    while (j < jmax)
    {
        j += 2;
        ret[k] *= ret[k - 1];
        sum += castTo<T>(twicePlusOne(j)) * ret[k] * ret[k];
        ++k;
    }
    //
    // Normalization
    //
    // Adjust sign factor so that sgn[f(jmax)] = (-1)^{j1+j2}.
    bool sign  = (jmin & 1) == static_cast<IntT>(ret[siz - 1] < T());
    auto scale = (sign ? pre : -pre) / std::sqrt(sum);
    auto djj   = castTo<T>(twicePlusOne(jmin));
    for (decltype(siz) i = 0; i < siz; ++i)
    {
        ret[i] *= std::sqrt(djj) * scale;
        djj += T(4);
    }

    return;
}

//
// Implementation body for Gaunt coefficient
//
template <typename T, typename ArgT>
struct gaunt_rec_m_impl
{
    using Scalar          = T;
    using ArgumentType    = ArgT;
    using IntegerType     = typename IntegerTypeOf<ArgT>::type;
    using impl_3j_rec_m_t = wigner3j_rec_m_impl<T, ArgT>;

    static std::tuple<ArgumentType, ArgumentType> args_range(ArgumentType j1,
                                                             ArgumentType j2,
                                                             ArgumentType j3,
                                                             ArgumentType m1)
    {
        return impl_3j_rec_m_t::args_range(j1, j2, j3, m1);
    }

    template <typename SizeT>
    static SizeT storage_size(ArgumentType jmin, ArgumentType jmax,
                              ArgumentType j1, ArgumentType j2, ArgumentType,
                              ArgumentType)
    {
        const auto n1 = castTo<SizeT>(jmax - jmin + 1);
        const auto n2 = castTo<SizeT>(j1 + j2 - abs(j1 - j2)) / 2 + 1;
        return std::max(n1, n2);
    }

    template <typename RandomAccessIterator>
    static void run(RandomAccessIterator out, ArgumentType mmin,
                    ArgumentType mmax, ArgumentType j1, ArgumentType j2,
                    ArgumentType j3, ArgumentType m1)
    {
        using difference_type = typename std::iterator_traits<
            RandomAccessIterator>::difference_type;

        detail::gaunt_pre(j1, j2, out);
        auto j0  = abs(j1 - j2);
        auto pre = out[static_cast<difference_type>((j3 - j0) / 2)];

        impl_3j_rec_m_t::run(out, mmin, mmax, j1, j2, j3, m1);
        auto last = out + static_cast<difference_type>(mmax - mmin + 1);

        *out *= pre;
        while (++out != last)
        {
            *out *= pre;
        }
    }
};

} // namespace: detail

/*!
 * Gaunt coefficient calculator.
 */
template <typename T, typename ArgT = int>
class GauntSeriesJ
{
    static_assert(std::is_integral<ArgT>::value && std::is_signed<ArgT>::value,
                  "Invalid type for template argument ArgT: "
                  "a signed integral type is expected");

public:
    using Scalar        = T;
    using ArgumentType  = ArgT;
    using IntegerType   = ArgT;
    using ContainerType = std::vector<T>;
    using Index         = typename ContainerType::size_type;

    explicit GauntSeriesJ(Index capacity = 51)
        : data_(capacity), argmin_(ArgumentType()), argmax_(ArgumentType(-1))
    {
    }

    GauntSeriesJ(const GauntSeriesJ&) = default;
    GauntSeriesJ(GauntSeriesJ&&)      = default;
    ~GauntSeriesJ()                   = default;

    GauntSeriesJ& operator=(const GauntSeriesJ&) = default;
    GauntSeriesJ& operator=(GauntSeriesJ&&) = default;

    Index size() const
    {
        return static_cast<Index>((argmax_ - argmin_) / 2 + 1);
    }

    ArgumentType nmin() const
    {
        return argmin_;
    }

    ArgumentType nmax() const
    {
        return argmax_;
    }

    void compute(ArgumentType j1, ArgumentType j2, ArgumentType m1,
                 ArgumentType m2);

    Scalar get(ArgumentType j3) const
    {
        if (nmin() <= j3 && j3 <= nmax())
        {
            auto pos2 = j3 - nmin();
            return pos2 % 2 ? Scalar() : data_[castTo<Index>(pos2 / 2)];
        }
        else
        {
            return Scalar();
        }
    }

    const Scalar& unsafeRef(ArgumentType j3) const
    {
        assert((j3 - nmin()) % 2 == 0 && nmin() <= j3 && j3 <= nmax());
        return data_[castTo<Index>(j3 - nmin()) / 2];
    }

    void shrinkToFit()
    {
        if (data_.size() != size())
        {
            ContainerType tmp(std::begin(data_), std::begin(data_) + size());
            data_ = std::move(tmp);
        }
    }

private:
    void expand_if_necessary(Index n)
    {
        if (n > data_.size())
        {
            data_.resize(n);
        }
    }

    Wigner3jSeriesJ<T, ArgumentType> w3j_;
    ContainerType data_;
    ArgumentType argmin_;
    ArgumentType argmax_;
};

template <typename T, typename ArgT>
void GauntSeriesJ<T, ArgT>::compute(ArgumentType j1, ArgumentType j2,
                                    ArgumentType m1, ArgumentType m2)
{
    if (!(detail::check_jm_pair(j1, m1) && detail::check_jm_pair(j2, m2)))
    {
        argmin_ = ArgumentType(-1);
        argmax_ = ArgumentType(-2);
        return;
    }

    argmin_ = abs(j1 - j2);
    argmax_ = j1 + j2;

    expand_if_necessary(size());

    detail::gaunt_pre(j1, j2, std::begin(data_));
    w3j_.compute(j1, j2, m1, m2);

    auto l0   = abs(j1 - j2);
    auto l    = w3j_.nmin();
    auto lmax = w3j_.nmax();
    if ((l + lmax) & 1)
    {
        ++l;
    }

    argmin_ = l;
    argmax_ = lmax;

    auto offset = (l - l0) / 2;
    auto it_pre = std::begin(data_) + offset;
    auto target = std::begin(data_);

    while (l <= lmax)
    {
        *target = (*it_pre) * w3j_.unsafeRef(l);
        l += 2;
        ++it_pre;
        ++target;
    }
}

///
/// Yet another Gaunt coefficient calculator.
///
template <typename T, typename ArgT = int>
using GauntSeriesM =
    WignerSeriesContainer<T, ArgT, detail::gaunt_rec_m_impl<T, ArgT>>;

//------------------------------------------------------------------------------
// Gaunt coefficient table
//------------------------------------------------------------------------------

///
/// ## GauntTable
///
/// Table of Gaunt coefficients
///
/// ### Description
///
/// The class provide a container that stores pre-computed values of Gaunt
/// coefficients `G(l1,l2,l3, m1,m2,m3)` following the indexing scheme proposed
/// by Pinchon and Hoggan.
///
template <typename T, typename ArgT = int>
class GauntTable
{
    static_assert(std::is_integral<ArgT>::value && std::is_signed<ArgT>::value,
                  "Invalid type for template argument ArgT: "
                  "a signed integral type is expected");

    using Evaluator = GauntSeriesM<T, ArgT>;

public:
    using Scalar       = typename Evaluator::Scalar;
    using ArgumentType = typename Evaluator::ArgumentType;
    using IntegerType  = typename Evaluator::IntegerType;

private:
    using VectorType     = std::vector<Scalar>;
    using VectorOfVector = std::vector<VectorType>;

public:
    using Index = typename VectorType::size_type;

    explicit GauntTable(ArgumentType argmax) : argmax_(), nnz_(), data_()
    {
        assert(argmax > ArgumentType());
        gen_table(argmax);
    }

    ~GauntTable() = default;

    /// @return maximum value of arguments
    ArgumentType nmax() const
    {
        return argmax_;
    }

    /// @return size of outer vector
    Index outerSize() const
    {
        return data_.size();
    }

    /// @return Number of non-zero coefficients stored in the table.
    Index nnz() const
    {
        return nnz_;
    }

    ///
    /// @return value of the Gaunt coefficient `G(l1,l2,l3, m1,m2,m3)` for given
    /// six arguments
    ///
    /// @pre `0 <= l && l <= nmax()` must be hold for all `l=l1,l2,l3`.
    ///  Otherwise, undefined behavior.
    ///
    Scalar operator()(ArgumentType l1, ArgumentType l2, ArgumentType l3,
                      ArgumentType m1, ArgumentType m2, ArgumentType m3) const
    {
        assert(0 <= l1 && l1 <= argmax_);
        assert(0 <= l2 && l2 <= argmax_);
        assert(0 <= l3 && l3 <= argmax_);
        //
        // Quick return in case of the value is trivially zero for given
        // arguments
        //
        if ((m1 + m2 + m3) != ArgumentType())
        {
            return Scalar();
        }

        if ((l1 + l2 + l3) % 2 != 0)
        {
            return Scalar();
        }

        if (!detail::check_triad(l1, l2, l3))
        {
            return Scalar();
        }

        if (!(detail::check_jm_pair(l1, m1) && detail::check_jm_pair(l2, m2) &&
              detail::check_jm_pair(l3, m3)))
        {
            return Scalar();
        }

        return coeffNoCheck(l1, l2, l3, m1, m2, m3);
    }

    ///
    /// Get the value of Gaunt coefficient `G(l1,l2,l3,m1,m2,m3)` without
    /// checking the range of arguments.
    ///
    /// @pre arguments must satisfy the following relations: otherwise,
    /// undefined behavior.
    ///
    ///  - 0 <= l1,l2,l3 <= nmax()
    ///  - |m1| <= l1, |m2| <= l2, |m3| <= l3
    ///  - l1 + l2 + l3: even
    ///  - m1 + m2 + m3 = 0
    ///  - (l1, l2, l3) satisfy the triangular condition,
    ///     i.e., |l1 - l2| <= l3 <= l2 + l3
    ///
    Scalar coeffNoCheck(ArgumentType l1, ArgumentType l2, ArgumentType l3,
                        ArgumentType m1, ArgumentType m2, ArgumentType m3) const
    {
        assert(0 <= l1 && l1 <= nmax());
        assert(0 <= l2 && l2 <= nmax());
        assert(0 <= l3 && l3 <= nmax());
        assert(std::abs(m1) <= l1);
        assert(std::abs(m2) <= l2);
        assert(std::abs(m3) <= l3);
        assert((l1 + l2 + l3) % 2 == 0);
        assert(m1 + m2 + m3 == 0);
        assert(detail::check_triad(l1, l2, l3));

        //
        // Since the Gaunt coefficient is unchanged by the permutation of
        // (l,m) pairs, we swap the arguments so as to ensure l3 <= l2 <= l1.
        //
        if (l1 < l2)
        {
            std::swap(l1, l2);
            std::swap(m1, m2);
        }
        if (l1 < l3)
        {
            std::swap(l1, l3);
            std::swap(m1, m3);
        }
        if (l2 < l3)
        {
            std::swap(l2, l3);
            std::swap(m2, m3);
        }
        //
        // If m3 >= 0, read the value stored in the table.
        // If m3 < 0, use the fact that Gaunt coefficient is invariant under the
        // change of the sign of all `m`:
        //
        // `G(l1,l2,l3, m1,m2,m3) = G(l1,l2,l3, -m1,-m2,-m3)`
        //
        return m3 >= ArgumentType()
                   ? data_[outer_index(l1, l2, l3, m3)][l2 + m2]
                   : data_[outer_index(l1, l2, l3, -m3)][l2 - m2];
    }

    // ///
    // /// @return const reference to the coefficient corresponding to the
    // /// given
    // /// arguments.
    // ///
    // /// @pre the arguments must be satisfy the following relations:
    // ///
    // ///  - `0 <= l1 <= nmax()`
    // ///  - `(l1 + 1) / 2 <= l2 <= l1`
    // ///  - `l1 - l2 <= l3 <= l2``
    // ///  - `|m2| <= l2`
    // ///  - ` 0 <= m3 <= l3`
    // ///  - `l1 + l2 + l3: even`
    // ///
    // /// @remark This function do not check the boundaries of arguments. If
    // /// arguments do not satisfy the prerequisite conditions above, the
    // /// behavior
    // /// is undefined.
    // ///
    // const Scalar& coeffRef(ArgumentType l1, ArgumentType l2, ArgumentType l3,
    //                        ArgumentType m2, ArgumentType m3) const
    // {
    //     assert(0 <= l1 && l1 <= argmax_);
    //     assert((l1 + 1) / 2 <= l2 && l2 <= l1);
    //     assert(l1 - l2 <= l3 && l3 <= l2);
    //     assert(std::abs(m2) <= l2);
    //     assert(0 <= m3 && m3 <= l3);
    //     assert((l1 + l2 + l3) % 2 == 0);

    //     return data_[outer_index(l1, l2, l3, m3)][l2 + m2];
    // }

    void swap(GauntTable& x)
    {
        std::swap(argmax_, x.argmax_);
        std::swap(nnz_, x.nnz_);
        std::swap(data_, x.data_);
    }

private:
    // --- private members
    void gen_table(ArgumentType argmax);

    // Get the outer index of the coefficient
    static Index outer_index(ArgumentType l1, ArgumentType l2, ArgumentType l3,
                             ArgumentType m3)
    {
        // Pinchon's storage scheme
        assert((l1 + 1) / 2 <= l2 && l2 <= l1);
        assert(l1 - l2 <= l3 && l3 <= l2);
        assert(0 <= m3 && m3 <= l3);

        Index n;
        if (l1 % 2 == 0) // in case l1 even
        {
            const Index t = l1 / 2;
            n             = t * (((3 * t + 16) * t - 3) * t - 4);
            n += 3 * l3 * l3 + 6 * l2 * (l1 + 1);
            n -= 3 * (l2 * (l1 + 1) * (l1 - l2));
            n /= 12;
            n += m3;
        }
        else // in case l1 odd
        {
            const Index t = (l1 - 1) / 2;
            n             = (((3 * t + 22) * t + 33) * t + 26) * t + 21;
            n += 3 * (l3 * l3 + l2 * l1);
            n -= 3 * (l2 * l1 * (l1 - l2));
            n /= 12;
            n += m3 - 1;
        }
        return n;
    }

    // Size of outer vector
    static Index ptr_size(ArgumentType l)
    {
        if (l % 2 == 0) // l even
        {
            Index m = static_cast<Index>(l / 2);
            return (((m + 1) * (m + 2)) / 2 * ((3 * m + 7) * m + 6)) / 6;
        }
        else // l odd
        {
            Index m = static_cast<Index>((l - 1) / 2);
            return (((m + 1) * (m + 2) * (m + 3)) / 6 * (3 * m + 4)) / 2;
        }
    }

    static Index nnz_rasch(ArgumentType l)
    {
        Index nnz;
        if (l % 2 == 0) // l even
        {
            Index m = static_cast<Index>(l / 2);
            nnz     = ((m + 1) * (m + 2)) / 2;
            nnz *= ((36 * m + 97) * m + 107) * m + 30;
            nnz /= 30;
        }
        else // l odd
        {
            Index m = static_cast<Index>((l - 1) / 2);
            nnz     = ((m + 1) * (m + 2) * (m + 3)) / 6;
            nnz *= (18 * l + 43) * l + 19;
            nnz /= 20;
        }
        return nnz;
    }

    // --- local variables
    ArgT argmax_;         // maximum value of l
    Index nnz_;           // number of non-zero coefficients
    VectorOfVector data_; // table of coefficients
};

template <typename T, typename ArgT>
void GauntTable<T, ArgT>::gen_table(ArgumentType argmax)
{
    assert(argmax > ArgumentType());

    auto nnz              = Index();
    const auto outer_size = ptr_size(argmax);
    VectorOfVector data(outer_size);

    Index outer_index = 0;

    Evaluator gaunt_m;
    for (auto l1 = ArgumentType(); l1 <= argmax; ++l1)
    {
        for (auto l2 = (l1 + 1) / 2; l2 <= l1; ++l2)
        {
            for (auto l3 = l1 - l2; l3 <= l2; ++l3)
            {
                if ((l1 + l2 + l3) % 2 == 1)
                {
                    continue;
                }
                for (auto m3 = ArgumentType(); m3 <= l3; ++m3)
                {
                    const auto m2_end     = std::min(l1 - m3, l2) + 1;
                    const auto inner_size = static_cast<Index>(l2 + m2_end);
                    data[outer_index].resize(inner_size);
                    // GauntSeriesM compute G(l1,l2,l3, m1,m,-m1-m) for all
                    // possible m.
                    gaunt_m.compute(l3, l2, l1, m3);
                    for (auto m2 = -l2; m2 < m2_end; ++m2)
                    {
                        data[outer_index][l2 + m2] = gaunt_m.get(m2);
                    }
                    nnz += inner_size;
                    ++outer_index;
                }
            }
        }
    }
    assert(outer_index == outer_size);
    assert(nnz == nnz_rasch(argmax));

    argmax_ = argmax;
    nnz_    = nnz;
    std::swap(data_, data);
}

//--------------------------------------------------------------------------
// Global function
//--------------------------------------------------------------------------

namespace detail
{

template <typename T>
const GauntTable<T, int>& init_gaunt_table()
{
    // Initialize a table of Gaunt coefficients using the "construct on first
    // use" idiom.
    static GauntTable<T, int> table(MAX_LVALUE);
    return table;
}

} // namespace detail
///
/// Get the Gaunt coefficient
///
/// The function returns the value of a Gaunt coefficient `G(l1,l2,l3,
/// m1,m2,m3)` corresponding to the given arguments, for l1, l2, and l3 from 0
/// upto `wigher::MAX_LVALUE`.
///
/// On first call, all the Gaunt coefficients for l upto `wigher::MAX_LVALUE`
/// are computed to generate the table, so it takes time. Later, the function
/// can return the value quickly by looking-up the table.
///
template <typename T>
T gaunt(int l1, int l2, int l3, int m1, int m2, int m3)
{
    static const GauntTable<T, int>& table = detail::init_gaunt_table<T>();

    return table(l1, l2, l3, m1, m2, m3);
}

///
/// Get the Gaunt coefficient without checking the range of arguments.
///
/// See `GauntTable::coeffNoCheck()` for the conditions to be satisfied.
///
template <typename T>
T gauntNoCheck(int l1, int l2, int l3, int m1, int m2, int m3)
{
    static const GauntTable<T, int>& table = detail::init_gaunt_table<T>();
    return table.coeffNoCheck(l1, l2, l3, m1, m2, m3);
}

} // namespace: wigner

#endif /* WIGNER_GAUNT */
