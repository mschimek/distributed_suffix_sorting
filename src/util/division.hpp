#pragma once


namespace dsss::util {

template <typename T1, typename T2>
inline T1 div_ceil(T1 a, T2 b) {
    return (a + b - 1) / b;
}

};
