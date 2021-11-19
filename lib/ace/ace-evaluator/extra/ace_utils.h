//
// Created by Yury Lysogorskiy on 27.02.20.
//

#ifndef ACE_UTILS_H
#define ACE_UTILS_H
#include <cmath>
#include <string>
#include <sstream>

#include "ace_types.h"

using namespace std;

inline int sign(DOUBLE_TYPE x) {
    if (x < 0) return -1;
    else if (x > 0) return +1;
    else return 0;
}

inline double absolute_relative_error(double x, double y, double zero_threshold = 5e-6) {
    if (x == 0 && y == 0) return 0;
    else if (x == 0 || y == 0) return (abs(x + y) < zero_threshold ? 0 : 2);
    else return 2 * abs(x - y) / (abs(x) + abs(y));
}

//https://stackoverflow.com/questions/9277906/stdvector-to-string-with-custom-delimiter
template <typename T>
string join(const T& v, const string& delim) {
    stringstream s;
    for (const auto& i : v) {
        if (&i != &v[0]) {
            s << delim;
        }
        s << i;
    }
    return s.str();
}


#endif //ACE_UTILS_H
