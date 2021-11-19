/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2017 Hidekazu Ikeno
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
/// \file binomial.hpp
///

#ifndef WIGNER_BINOMIAL_HPP
#define WIGNER_BINOMIAL_HPP

#include <cassert>
#include <cmath>
#include <vector>

namespace wigner
{

///
/// Table of binomial coefficients \f$\binom{n}{k}\f$
///
template <typename T>
class BinomialTable
{
public:
    using Scalar    = T;
    using Container = std::vector<T>;
    using SizeType  = typename Container::size_type;

private:
    SizeType m_nmax;
    Container m_data;

public:
    BinomialTable() : m_nmax(0), m_data({T(1)})
    {
    }

    BinomialTable(SizeType nmax) : m_nmax(0), m_data()
    {
        resize(nmax);
    }

    BinomialTable(const BinomialTable&) = default;
    BinomialTable(BinomialTable&&)      = default;
    ~BinomialTable()                    = default;

    BinomialTable& operator=(const BinomialTable&) = default;
    BinomialTable& operator=(BinomialTable&&) = default;

    void resize(SizeType nmax);

    Scalar operator()(SizeType n, SizeType k) const
    {
        assert(k <= n && n <= max_arg());
        return m_data[index(n, k)];
    }

    SizeType max_arg() const
    {
        return m_nmax;
    }

private:
    static SizeType index(SizeType n, SizeType k)
    {
        assert(k <= n);
        return n * (n + 1) / 2 + k;
    }
};

template <typename T>
void BinomialTable<T>::resize(SizeType nmax)
{
    Container tmp(index(nmax + 1, nmax + 1));

    auto it = tmp.begin();
    for (SizeType n = 0; n <= nmax; ++n)
    {
        auto x = Scalar(1); // binomoal(n, 0) = 1
        *it    = x;
        ++it;
        for (SizeType k = 1; k <= n; ++k)
        {
            x *= Scalar(n - k + 1);
            x /= Scalar(k);
            *it = x;
            ++it;
        }
    }

    m_nmax = nmax;
    m_data.swap(tmp);
}

///
/// Table of square roots of binomial coefficients \f$\binom{n}{k}&^{1/2}\f$
///
template <typename T>
class SqrtBinomialTable
{
public:
    using Scalar    = T;
    using Container = std::vector<Scalar>;
    using SizeType  = typename Container::size_type;

private:
    int m_nmax;
    Container m_data;

public:
    SqrtBinomialTable() : m_nmax(0), m_data({Scalar(1)})
    {
    }

    SqrtBinomialTable(int nmax) : m_nmax(0), m_data()
    {
        resize(nmax);
    }

    SqrtBinomialTable(const SqrtBinomialTable&) = default;
    SqrtBinomialTable(SqrtBinomialTable&&)      = default;
    ~SqrtBinomialTable()                        = default;

    SqrtBinomialTable& operator=(const SqrtBinomialTable&) = default;
    SqrtBinomialTable& operator=(SqrtBinomialTable&&) = default;

    void resize(int nmax);

    Scalar operator()(int n, int k) const
    {
        assert(n <= max_arg());
        return (n >= 0 && k <= n) ? m_data[index(n, k)] : Scalar();
    }

    int max_arg() const
    {
        return m_nmax;
    }

private:
    static SizeType index(int n, int k)
    {
        assert(k <= n);
        return static_cast<SizeType>(n * (n + 1) / 2 + k);
    }
};

template <typename T>
void SqrtBinomialTable<T>::resize(int nmax)
{
    if (nmax < 0)
        return;

    Container tmp(index(nmax + 1, nmax + 1));

    auto it = tmp.begin();
    for (int n = 0; n <= nmax; ++n)
    {
        auto x = Scalar(1); // binomoal(n, 0) = 1
        *it    = x;
        ++it;
        for (int k = 1; k <= n; ++k)
        {
            x *= std::sqrt(Scalar(n - k + 1) / Scalar(k));
            *it = x;
            ++it;
        }
    }

    m_nmax = nmax;
    m_data.swap(tmp);
}

} // namespace: wigner

#endif /* WIGNER_BINOMIAL_HPP */
