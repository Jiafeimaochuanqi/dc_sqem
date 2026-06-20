#ifndef __ACCUMARRAY_H__
#define __ACCUMARRAY_H__

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

// Header-only C++ implementation of the Python functions:
//
// def accumarray(S, V):
//     assert S.size == V.size
//     if S.size == 0: return np.zeros((0, 1))
//     A = np.zeros((S.max() + 1, 1))
//     for s in range(S.size): A[S[s]] += V[s]
//     return A
//
// def accumarray_scalar(S, V):
//     if S.size == 0: return np.zeros((0, 1))
//     A = np.zeros((S.max() + 1, 1))
//     for s in range(S.size): A[S[s]] += V
//     return A
//
// Notes:
// - S must contain non-negative integer indices.
// - The output is always an Eigen column vector of shape (max(S)+1, 1).
// - Empty S returns a 0-row column vector, matching np.zeros((0, 1)).

// Sum values V[i] into A[S[i]].
// S: integer Eigen vector / array expression, shape (N,) or (N,1).
// V: scalar Eigen vector / array expression with the same number of elements.
// Return: Eigen::Matrix<Scalar, Dynamic, 1>.
/**
 * sum according group indices
 */
template <typename IndexDerived, typename ValueDerived>
inline auto accumarray(
    const Eigen::MatrixBase<IndexDerived>& S,
    const Eigen::MatrixBase<ValueDerived>& V)
    -> Eigen::Matrix<typename ValueDerived::Scalar, Eigen::Dynamic, 1>
{
    using IndexScalar = typename IndexDerived::Scalar;
    using ValueScalar = typename ValueDerived::Scalar;

    static_assert(std::is_integral<IndexScalar>::value,
                  "accumarray: S must contain integer indices.");

    if (S.size() != V.size()) {
        throw std::invalid_argument("accumarray: S and V must have the same size.");
    }

    if (S.size() == 0) {
        return Eigen::Matrix<ValueScalar, Eigen::Dynamic, 1>(0, 1);
    }

    const auto max_index = S.maxCoeff();
    const auto min_index = S.minCoeff();

    if (min_index < 0) {
        throw std::invalid_argument("accumarray: S must contain non-negative indices.");
    }

    Eigen::Matrix<ValueScalar, Eigen::Dynamic, 1> A(
        static_cast<Eigen::Index>(max_index) + 1,
        1
    );
    A.setZero();

    for (Eigen::Index i = 0; i < S.size(); ++i) {
        A(static_cast<Eigen::Index>(S.derived().coeff(i))) += V.derived().coeff(i);
    }

    return A;
}

// Sum the same scalar value into A[S[i]] for every i.
// S: integer Eigen vector / array expression, shape (N,) or (N,1).
// value: scalar to add for every index.
// Return: Eigen::Matrix<Scalar, Dynamic, 1>.
template <typename IndexDerived, typename ValueScalar>
inline auto accumarray_scalar(
    const Eigen::MatrixBase<IndexDerived>& S,
    const ValueScalar& value)
    -> Eigen::Matrix<ValueScalar, Eigen::Dynamic, 1>
{
    using IndexScalar = typename IndexDerived::Scalar;

    static_assert(std::is_integral<IndexScalar>::value,
                  "accumarray_scalar: S must contain integer indices.");

    if (S.size() == 0) {
        return Eigen::Matrix<ValueScalar, Eigen::Dynamic, 1>(0, 1);
    }

    const auto max_index = S.maxCoeff();
    const auto min_index = S.minCoeff();

    if (min_index < 0) {
        throw std::invalid_argument("accumarray_scalar: S must contain non-negative indices.");
    }

    Eigen::Matrix<ValueScalar, Eigen::Dynamic, 1> A(
        static_cast<Eigen::Index>(max_index) + 1,
        1
    );
    A.setZero();

    for (Eigen::Index i = 0; i < S.size(); ++i) {
        A(static_cast<Eigen::Index>(S.derived().coeff(i))) += value;
    }

    return A;
}

// Convenience overloads for common dynamic Eigen vector types.
inline Eigen::VectorXd accumarray(
    const Eigen::VectorXi& S,
    const Eigen::VectorXd& V)
{
    return accumarray<Eigen::VectorXi, Eigen::VectorXd>(S, V);
}

inline Eigen::VectorXf accumarray(
    const Eigen::VectorXi& S,
    const Eigen::VectorXf& V)
{
    return accumarray<Eigen::VectorXi, Eigen::VectorXf>(S, V);
}

inline Eigen::VectorXd accumarray_scalar(
    const Eigen::VectorXi& S,
    double value)
{
    return accumarray_scalar<Eigen::VectorXi, double>(S, value);
}

inline Eigen::VectorXf accumarray_scalar(
    const Eigen::VectorXi& S,
    float value)
{
    return accumarray_scalar<Eigen::VectorXi, float>(S, value);
}

#endif // __ACCUMARRAY_H__