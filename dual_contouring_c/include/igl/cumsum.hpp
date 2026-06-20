#ifndef __CUMSUM_H__
#define __CUMSUM_H__
#include <Eigen/Dense>
#include <stdexcept>
#include <type_traits>

// Header-only C++ implementation of the Python function:
//
// def cumsum(X, dim, zero_prefix=False):
//     if dim == 1:
//         Y = np.cumsum(X, axis=0)
//         if zero_prefix:
//             Y = np.vstack([np.zeros((1, X.shape[1])), Y])
//     elif dim == 2:
//         Y = np.cumsum(X, axis=1)
//         if zero_prefix:
//             Y = np.hstack([np.zeros((X.shape[0], 1)), Y])
//     else:
//         raise ValueError("dim must be 1 (columns) or 2 (rows)")
//     return Y
//
// Notes:
// - dim == 1: cumulative sum down rows, i.e. along axis 0.
// - dim == 2: cumulative sum across columns, i.e. along axis 1.
// - zero_prefix == true adds a leading zero row for dim == 1,
//   or a leading zero column for dim == 2.
// - Output scalar type is the same as X's scalar type.

template <typename Derived>
inline auto cumsum_impl(
    const Eigen::MatrixBase<Derived>& X,
    int dim,
    bool zero_prefix)
    -> Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
{
    using Scalar = typename Derived::Scalar;
    using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    const Eigen::Index rows = X.rows();
    const Eigen::Index cols = X.cols();

    if (dim == 1) {
        MatrixX Y(rows + (zero_prefix ? 1 : 0), cols);

        if (zero_prefix) {
            Y.row(0).setZero();
            for (Eigen::Index r = 0; r < rows; ++r) {
                if (r == 0) {
                    Y.row(1) = X.row(0);
                } else {
                    Y.row(r + 1) = Y.row(r) + X.row(r);
                }
            }
        } else {
            if (rows == 0) {
                Y.resize(0, cols);
                return Y;
            }

            Y.row(0) = X.row(0);
            for (Eigen::Index r = 1; r < rows; ++r) {
                Y.row(r) = Y.row(r - 1) + X.row(r);
            }
        }

        return Y;
    }

    if (dim == 2) {
        MatrixX Y(rows, cols + (zero_prefix ? 1 : 0));

        if (zero_prefix) {
            Y.col(0).setZero();
            for (Eigen::Index c = 0; c < cols; ++c) {
                if (c == 0) {
                    Y.col(1) = X.col(0);
                } else {
                    Y.col(c + 1) = Y.col(c) + X.col(c);
                }
            }
        } else {
            if (cols == 0) {
                Y.resize(rows, 0);
                return Y;
            }

            Y.col(0) = X.col(0);
            for (Eigen::Index c = 1; c < cols; ++c) {
                Y.col(c) = Y.col(c - 1) + X.col(c);
            }
        }

        return Y;
    }

    throw std::invalid_argument("cumsum: dim must be 1 (axis 0 / rows) or 2 (axis 1 / columns).");
}


// General Eigen matrix/expression overload.
template <typename Derived>
inline auto cumsum(
    const Eigen::MatrixBase<Derived>& X,
    int dim,
    bool zero_prefix = false)
    -> Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
{
    return cumsum_impl(X, dim, zero_prefix);
}

// Convenience overloads for common dynamic matrix types.
inline Eigen::MatrixXd cumsum(
    const Eigen::MatrixXd& X,
    int dim,
    bool zero_prefix = false)
{
    return cumsum_impl(X, dim, zero_prefix);
}

inline Eigen::MatrixXf cumsum(
    const Eigen::MatrixXf& X,
    int dim,
    bool zero_prefix = false)
{
    return cumsum_impl(X, dim, zero_prefix);
}

inline Eigen::MatrixXi cumsum(
    const Eigen::MatrixXi& X,
    int dim,
    bool zero_prefix = false)
{
    return cumsum_impl(X, dim, zero_prefix);
}

#endif // __CUMSUM_H__