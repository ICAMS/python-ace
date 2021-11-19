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
/// @file wigner/constants.hpp
///

#ifndef WIGNER_CONSTANTS_HPP
#define WIGNER_CONSTANTS_HPP

namespace wigner
{

/* Maximum values for the arguments determined at compile time */
#ifndef WIGNER_LMAX_VALUE
#define WIGNER_LMAX_VALUE 40
#endif

constexpr const int MAX_LVALUE = WIGNER_LMAX_VALUE;

///
/// Mathematical constants for internal use
///
template <typename T>
struct math_const
{
    constexpr static T pi = T(
        3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679);
    constexpr static T sqrt_pi = T(
        1.7724538509055160272981674833411451827975494561223871282138077898529112845910321813749506567385446654);
};
} // namespace wigner

#endif /* WIGNER_CONSTANTS_HPP */
