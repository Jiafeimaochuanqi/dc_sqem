#ifndef __DOUBLEAREA_H__
#define __DOUBLEAREA_H__
#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

// Header-only C++ implementation of doublearea.py
//
// Python behavior summary:
// - doublearea(V,F):
//   * if F has 4 columns, split each quad into two triangles and sum their double areas.
//   * if V is 3D, return ||(V1-V0) x (V2-V0)|| for each triangle, i.e. 2*area.
//   * if V is 2D, return signed double area for each triangle.
// - doublearea_single(A,B,C): return 3D triangle double area.
// - doublearea_from_edge_lengths(ul): Kahan/Heron formula for triangle double area.
//
// Notes:
// - Face indices are assumed to be 0-based, matching Python / NumPy.
// - Triangle faces must have 3 columns; quad faces must have 4 columns.
// - V must have 2 or 3 columns.
// - Output is an Eigen column vector.


template <typename FaceDerived>
inline void validate_face_indices(
    const Eigen::MatrixBase<FaceDerived>& F,
    Eigen::Index num_vertices)
{
    using FaceScalar = typename FaceDerived::Scalar;
    static_assert(std::is_integral<FaceScalar>::value,
                  "doublearea: F must contain integer vertex indices.");

    for (Eigen::Index i = 0; i < F.rows(); ++i) {
        for (Eigen::Index j = 0; j < F.cols(); ++j) {
            const auto idx = F.derived().coeff(i, j);
            if (idx < 0 || idx >= num_vertices) {
                throw std::out_of_range("doublearea: face index out of range.");
            }
        }
    }
}

template <typename VertexDerived, typename FaceDerived>
inline auto doublearea_tri_impl(
    const Eigen::MatrixBase<VertexDerived>& V,
    const Eigen::MatrixBase<FaceDerived>& F)
    -> Eigen::Matrix<typename VertexDerived::Scalar, Eigen::Dynamic, 1>
{
    using Scalar = typename VertexDerived::Scalar;
    using FaceScalar = typename FaceDerived::Scalar;

    static_assert(std::is_integral<FaceScalar>::value,
                  "doublearea: F must contain integer vertex indices.");

    if (F.cols() != 3) {
        throw std::invalid_argument("doublearea_tri: F must have 3 columns.");
    }
    if (V.cols() != 2 && V.cols() != 3) {
        throw std::invalid_argument("doublearea_tri: V must have 2 or 3 columns.");
    }

    validate_face_indices(F, V.rows());

    const Eigen::Index m = F.rows();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dblA(m);
    dblA.setZero();

    if (V.cols() == 3) {
        for (Eigen::Index f = 0; f < m; ++f) {
            const Eigen::Index i0 = static_cast<Eigen::Index>(F.derived().coeff(f, 0));
            const Eigen::Index i1 = static_cast<Eigen::Index>(F.derived().coeff(f, 1));
            const Eigen::Index i2 = static_cast<Eigen::Index>(F.derived().coeff(f, 2));

            const Eigen::Matrix<Scalar, 3, 1> a = V.derived().row(i0).transpose();
            const Eigen::Matrix<Scalar, 3, 1> b = V.derived().row(i1).transpose();
            const Eigen::Matrix<Scalar, 3, 1> c = V.derived().row(i2).transpose();

            dblA(f) = (b - a).cross(c - a).norm();
        }
    } else { // V.cols() == 2, signed double area
        for (Eigen::Index f = 0; f < m; ++f) {
            const Eigen::Index i0 = static_cast<Eigen::Index>(F.derived().coeff(f, 0));
            const Eigen::Index i1 = static_cast<Eigen::Index>(F.derived().coeff(f, 1));
            const Eigen::Index i2 = static_cast<Eigen::Index>(F.derived().coeff(f, 2));

            const Scalar rx = V.derived().coeff(i0, 0) - V.derived().coeff(i2, 0);
            const Scalar sx = V.derived().coeff(i1, 0) - V.derived().coeff(i2, 0);
            const Scalar ry = V.derived().coeff(i0, 1) - V.derived().coeff(i2, 1);
            const Scalar sy = V.derived().coeff(i1, 1) - V.derived().coeff(i2, 1);

            dblA(f) = rx * sy - ry * sx;
        }
    }

    return dblA;
}

template <typename VertexDerived, typename FaceDerived>
inline auto doublearea_quad_impl(
    const Eigen::MatrixBase<VertexDerived>& V,
    const Eigen::MatrixBase<FaceDerived>& F)
    -> Eigen::Matrix<typename VertexDerived::Scalar, Eigen::Dynamic, 1>
{
    using Scalar = typename VertexDerived::Scalar;
    using FaceScalar = typename FaceDerived::Scalar;

    static_assert(std::is_integral<FaceScalar>::value,
                  "doublearea_quad: F must contain integer vertex indices.");

    if (F.cols() != 4) {
        throw std::invalid_argument("doublearea_quad: F must have 4 columns.");
    }

    if (V.cols() != 2 && V.cols() != 3) {
        throw std::invalid_argument("doublearea_quad: V must have 2 or 3 columns.");
    }

    validate_face_indices(F, V.rows());

    const Eigen::Index m = F.rows();

    Eigen::Matrix<int, Eigen::Dynamic, 3> Ft(2 * m, 3);

    for (Eigen::Index i = 0; i < m; ++i) {
        // Match Python:
        // Ft[2*i]     = [F[i,0], F[i,1], F[i,2]]
        // Ft[2*i + 1] = [F[i,2], F[i,3], F[i,0]]
        Ft(2 * i, 0) = static_cast<int>(F.derived().coeff(i, 0));
        Ft(2 * i, 1) = static_cast<int>(F.derived().coeff(i, 1));
        Ft(2 * i, 2) = static_cast<int>(F.derived().coeff(i, 2));

        Ft(2 * i + 1, 0) = static_cast<int>(F.derived().coeff(i, 2));
        Ft(2 * i + 1, 1) = static_cast<int>(F.derived().coeff(i, 3));
        Ft(2 * i + 1, 2) = static_cast<int>(F.derived().coeff(i, 0));
    }

    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> triA =
        doublearea_tri_impl(V, Ft);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dblA(m);

    for (Eigen::Index i = 0; i < m; ++i) {
        dblA(i) = triA(2 * i) + triA(2 * i + 1);
    }

    return dblA;
}

template <typename VertexDerived, typename FaceDerived>
inline auto doublearea(
    const Eigen::MatrixBase<VertexDerived>& V,
    const Eigen::MatrixBase<FaceDerived>& F)
    -> Eigen::Matrix<typename VertexDerived::Scalar, Eigen::Dynamic, 1>
{
    using FaceScalar = typename FaceDerived::Scalar;

    static_assert(std::is_integral<FaceScalar>::value,
                  "doublearea: F must contain integer vertex indices.");

    if (F.cols() == 4) {
        return doublearea_quad_impl(V, F);
    }

    if (F.cols() == 3) {
        return doublearea_tri_impl(V, F);
    }

    throw std::invalid_argument("doublearea: F must have 3 or 4 columns.");
}

template <typename VertexDerived, typename FaceDerived>
inline auto doublearea_quad(
    const Eigen::MatrixBase<VertexDerived>& V,
    const Eigen::MatrixBase<FaceDerived>& F)
    -> Eigen::Matrix<typename VertexDerived::Scalar, Eigen::Dynamic, 1>
{
    return doublearea_quad_impl(V, F);
}
// Return 3D triangle double area ||(B-A) x (C-A)||.
template <typename DerivedA, typename DerivedB, typename DerivedC>
inline auto doublearea_single(
    const Eigen::MatrixBase<DerivedA>& A,
    const Eigen::MatrixBase<DerivedB>& B,
    const Eigen::MatrixBase<DerivedC>& C)
    -> typename DerivedA::Scalar
{
    using Scalar = typename DerivedA::Scalar;

    if (A.size() != 3 || B.size() != 3 || C.size() != 3) {
        throw std::invalid_argument("doublearea_single: A, B, C must be 3D vectors.");
    }

    const Eigen::Matrix<Scalar, 3, 1> a = A.derived();
    const Eigen::Matrix<Scalar, 3, 1> b = B.derived();
    const Eigen::Matrix<Scalar, 3, 1> c = C.derived();

    return (b - a).cross(c - a).norm();
}

// Compute triangle double area from edge lengths using Kahan's Heron formula.
// ul: m x 3 matrix; each row contains the three edge lengths of one triangle.
// nan_replacement: replacement value if numerical invalidity produces NaN.
template <typename LengthDerived>
inline auto doublearea_from_edge_lengths(
    const Eigen::MatrixBase<LengthDerived>& ul,
    typename LengthDerived::Scalar nan_replacement = std::numeric_limits<typename LengthDerived::Scalar>::quiet_NaN())
    -> Eigen::Matrix<typename LengthDerived::Scalar, Eigen::Dynamic, 1>
{
    using Scalar = typename LengthDerived::Scalar;

    if (ul.cols() != 3) {
        throw std::invalid_argument("doublearea_from_edge_lengths: ul must have 3 columns.");
    }

    const Eigen::Index m = ul.rows();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dblA(m);
    dblA.setZero();

    for (Eigen::Index i = 0; i < m; ++i) {
        Scalar l0 = ul.derived().coeff(i, 0);
        Scalar l1 = ul.derived().coeff(i, 1);
        Scalar l2 = ul.derived().coeff(i, 2);

        if (l0 > l1) std::swap(l0, l1);
        if (l1 > l2) std::swap(l1, l2);
        if (l0 > l1) std::swap(l0, l1);

        const Scalar arg =
            (l0 + (l1 + l2)) *
            (l2 - (l0 - l1)) *
            (l2 + (l0 - l1)) *
            (l0 + (l1 - l2));

        Scalar value = static_cast<Scalar>(0.5) * std::sqrt(arg);

        if (std::isnan(static_cast<double>(value))) {
            value = nan_replacement;
        }

        if (std::isnan(static_cast<double>(value))) {
            throw std::runtime_error("DOUBLEAREA() PRODUCED NaN");
        }

        dblA(i) = value;
    }

    return dblA;
}

// Convenience overloads for common dynamic Eigen matrix types.
inline Eigen::VectorXd doublearea(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    return doublearea<Eigen::MatrixXd, Eigen::MatrixXi>(V, F);
}

inline Eigen::VectorXf doublearea(const Eigen::MatrixXf& V, const Eigen::MatrixXi& F)
{
    return doublearea<Eigen::MatrixXf, Eigen::MatrixXi>(V, F);
}

inline double doublearea_single(
    const Eigen::Vector3d& A,
    const Eigen::Vector3d& B,
    const Eigen::Vector3d& C)
{
    return doublearea_single<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>(A, B, C);
}

inline float doublearea_single(
    const Eigen::Vector3f& A,
    const Eigen::Vector3f& B,
    const Eigen::Vector3f& C)
{
    return doublearea_single<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>(A, B, C);
}

inline Eigen::VectorXd doublearea_from_edge_lengths(
    const Eigen::MatrixXd& ul,
    double nan_replacement = std::numeric_limits<double>::quiet_NaN())
{
    return doublearea_from_edge_lengths<Eigen::MatrixXd>(ul, nan_replacement);
}

inline Eigen::VectorXf doublearea_from_edge_lengths(
    const Eigen::MatrixXf& ul,
    float nan_replacement = std::numeric_limits<float>::quiet_NaN())
{
    return doublearea_from_edge_lengths<Eigen::MatrixXf>(ul, nan_replacement);
}


#endif // __DOUBLEAREA_H__