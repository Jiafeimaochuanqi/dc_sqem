#ifndef __NUMERICUTILS_H__
#define __NUMERICUTILS_H__
template<typename T>
int compareDist(T a, T b)
{
    const bool aInf = std::isinf(a);
    const bool bInf = std::isinf(b);

    if (aInf && bInf) return 0;
    if (aInf) return 1;   // inf > finite
    if (bInf) return -1;  // finite < inf
    const T absTol = std::is_same_v<T, float> ? static_cast<T>(1e-6f) : static_cast<T>(1e-12);
    const T relTol = std::is_same_v<T, float> ? static_cast<T>(1e-5f) : static_cast<T>(1e-10);

    T tol = absTol + relTol * std::max(std::abs(a), std::abs(b));

    if (a < b - tol) return -1;
    if (a > b + tol) return 1;
    return 0;
}
#endif // __NUMERICUTILS_H__