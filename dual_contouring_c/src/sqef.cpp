#include "sqef.h"
#include "mesh_utils.hpp"
SQEF::SQEF(const MatF3& normals , const MatF3& shape_samples,const MatF3& cell_samples)
:normals(normals),shape_samples(shape_samples),cell_samples(cell_samples)
{
    Eigen::VectorXf fixed_column = Eigen::VectorXf::Ones(normals.rows(), 1);
    A.resize(normals.rows(), normals.cols() + 1);
    A << normals, fixed_column;
    b = (normals.array() * shape_samples.array()).rowwise().sum();
}
std::optional<Eigen::Vector4f> SQEF::solve_with_group(std::vector<std::vector<int>>&groups,bool is_cylinder, double svd_threshold,bool debug)
{
    if (A.rows() < 4) {
        int original_rows = A.rows();
        int original_cols = A.cols();
        A.conservativeResize(original_rows * 2, original_cols);
        A.bottomRows(original_rows) = A.topRows(original_rows);
        b.conservativeResize(original_rows * 2);
        b.tail(original_rows) = b.head(original_rows);
    }
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);

    Eigen::MatrixXf Q = qr.householderQ();//MxM
    Eigen::VectorXf Qb = (Q.transpose() * b).topRows(4);
    Eigen::MatrixXf R = qr.matrixQR().triangularView<Eigen::Upper>();//Mx4
    R.conservativeResize(4, 4);//R-->R_{top4}


    if (debug) {
        std::cout << "R:\n" << R << "\n";
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXf s = svd.singularValues();
    Eigen::MatrixXf U = svd.matrixU();
    Eigen::MatrixXf Vt = svd.matrixV().transpose();

    if (debug) {
        std::cout << "s: " << s.transpose() << "\n";
        std::cout << "Vt:\n" << Vt << "\n";
    }

    for (int i = 1; i < s.size(); ++i) {
        if (s[i] < s[0] * svd_threshold) {
            if (i == 1) {
                Eigen::Vector3f v1 = Vt.row(1).head<3>().stableNormalized();
                Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
                dir = v1.cross(v3);
                dir.stableNormalize();
                is_plane = true;
                if(groups.size()>=2){
                    Eigen::Vector3f v2 = Vt.row(2).head<3>().stableNormalized();
                    Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
                    dir = v2.cross(v3);
                    dir.stableNormalize();
                    is_plane = true;
                    break;
                }else{
                    return std::nullopt;
                }
            } else if (i == 2) {
                Eigen::Vector3f v2 = Vt.row(2).head<3>().stableNormalized();
                Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
                dir = v2.cross(v3);
                dir.stableNormalize();
                is_plane = true;
                break;
            } else if (i == 3) {
                dir = Vt.row(3).head<3>().stableNormalized();
                is_line = true;
                break;
            }
        }
    }
    if(is_cylinder){
        is_plane=false;
        Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
        is_line=true;               
        dir=v3;
    }else{
        if(groups.size()==1&&is_plane){
            Eigen::Vector3f v1 = Vt.row(1).head<3>().stableNormalized();
            Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
            dir = v1.cross(v3);
            dir.stableNormalize();
            is_plane = true;
            return std::nullopt;
        }
        if (groups.size()<=2 && is_line)
        {
            if(groups.size()==2){
                is_line=false;
                is_plane=true;
                Eigen::Vector3f v2 = Vt.row(2).head<3>().stableNormalized();
                Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
                dir = v2.cross(v3);
                dir.stableNormalize();
            }else{
                is_line=false;
                is_plane=true;
                return std::nullopt;
            }
        }
        if(groups.size()<4 && !is_plane && !is_line){
            if(groups.size()==3){
                is_line=true;
                Eigen::Vector3f v3=Vt.row(3).head<3>().stableNormalized();                       
                dir=v3;
            }
            else if(groups.size()==2){
                is_plane=true;
                Eigen::Vector3f v2 = Vt.row(2).head<3>().stableNormalized();
                Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
                dir = v2.cross(v3);
                dir.stableNormalize();
            }
            else{
                is_line=false;
                is_plane=false;
                return std::nullopt;
            }
        }
    }

    Eigen::MatrixXf S_inv = Eigen::MatrixXf::Zero(4, 4);
    for (int i = 0; i < 4; ++i) {
        if (s[i] > 1e-6) {
            S_inv(i, i) = 1.0 / s[i];
        }
    }

    Eigen::MatrixXf A_inv = svd.matrixV() * S_inv * U.transpose();
    point = A_inv * Qb;
    return point;
}
std::optional<Eigen::Vector4f> SQEF::solve_in_corner(bool debug)
{
    if (A.rows() < 4) {

        int original_rows = A.rows();
        int original_cols = A.cols();

        A.conservativeResize(original_rows * 2, original_cols);
        A.bottomRows(original_rows) = A.topRows(original_rows);

        b.conservativeResize(original_rows * 2);
        b.tail(original_rows) = b.head(original_rows);
    }

    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);

    Eigen::MatrixXf Q = qr.householderQ();//MxM
    Eigen::VectorXf Qb = (Q.transpose() * b).topRows(4);
    Eigen::MatrixXf R = qr.matrixQR().triangularView<Eigen::Upper>();//Mx4
    R.conservativeResize(4, 4);//R-->R_{top4}
    

    if (debug) {
        std::cout << "R:\n" << R << "\n";
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXf s = svd.singularValues();
    Eigen::MatrixXf U = svd.matrixU();
    Eigen::MatrixXf Vt = svd.matrixV().transpose();

    if (debug) {
        std::cout << "s: " << s.transpose() << "\n";
        std::cout << "Vt:\n" << Vt << "\n";
    }
    Eigen::MatrixXf S_inv = Eigen::MatrixXf::Zero(4, 4);
    for (int i = 0; i < 4; ++i) {
        if (s[i] > 1e-6) {
            S_inv(i, i) = 1.0 / s[i];
        }
    }
    Eigen::MatrixXf A_inv = svd.matrixV() * S_inv * U.transpose();
    point = A_inv * Qb;
    return point;
}
std::optional<Eigen::Vector4f> SQEF::solve_in_line(double svd_threshold,bool debug)
{
    if (A.rows() < 4) {

        int original_rows = A.rows();
        int original_cols = A.cols();

        A.conservativeResize(original_rows * 2, original_cols);
        A.bottomRows(original_rows) = A.topRows(original_rows);

        b.conservativeResize(original_rows * 2);
        b.tail(original_rows) = b.head(original_rows);
    }

    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);

    Eigen::MatrixXf Q = qr.householderQ();//MxM
    Eigen::VectorXf Qb = (Q.transpose() * b).topRows(4);
    Eigen::MatrixXf R = qr.matrixQR().triangularView<Eigen::Upper>();//Mx4
    R.conservativeResize(4, 4);//R-->R_{top4}
    

    if (debug) {
        std::cout << "R:\n" << R << "\n";
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXf s = svd.singularValues();
    Eigen::MatrixXf U = svd.matrixU();
    Eigen::MatrixXf Vt = svd.matrixV().transpose();

    if (debug) {
        std::cout << "s: " << s.transpose() << "\n";
        std::cout << "Vt:\n" << Vt << "\n";
    }
    for (int i = 1; i < s.size(); ++i) {
        if (s[i] < s[0] * svd_threshold) {
            if (i == 1) {
                Eigen::Vector3f v1 = Vt.row(1).head<3>().stableNormalized();
                Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
                dir = v1.cross(v3);
                dir.stableNormalize();
                is_plane = true;
                return std::nullopt;
            } else if (i == 2) {
                Eigen::Vector3f v2 = Vt.row(2).head<3>().stableNormalized();
                Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
                dir = v2.cross(v3);
                dir.stableNormalize();
                is_plane = true;
                break;
            } else if (i == 3) {
                dir = Vt.row(3).head<3>().stableNormalized();
                is_line = true;
                break;
            }
        }
    }

    Eigen::MatrixXf S_inv = Eigen::MatrixXf::Zero(4, 4);
    for (int i = 0; i < 4; ++i) {
        if (s[i] > 1e-6) {
            S_inv(i, i) = 1.0 / s[i];
        }
    }
    if (is_plane){
        is_line=true;  
        is_plane=false;  
        Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();   
        dir=v3;
    }
    Eigen::MatrixXf A_inv = svd.matrixV() * S_inv * U.transpose();
    point = A_inv * Qb;
    return point;
}
std::optional<Eigen::Vector4f> SQEF::solve_in_plane(bool debug) {

    if (A.rows() < 4) {

        int original_rows = A.rows();
        int original_cols = A.cols();

        A.conservativeResize(original_rows * 2, original_cols);
        A.bottomRows(original_rows) = A.topRows(original_rows);

        b.conservativeResize(original_rows * 2);
        b.tail(original_rows) = b.head(original_rows);
    }

    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);

    Eigen::MatrixXf Q = qr.householderQ();//MxM
    Eigen::VectorXf Qb = (Q.transpose() * b).topRows(4);
    Eigen::MatrixXf R = qr.matrixQR().triangularView<Eigen::Upper>();//Mx4
    R.conservativeResize(4, 4);//R-->R_{top4}
    

    if (debug) {
        std::cout << "R:\n" << R << "\n";
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXf s = svd.singularValues();
    Eigen::MatrixXf U = svd.matrixU();
    Eigen::MatrixXf Vt = svd.matrixV().transpose();

    if (debug) {
        std::cout << "s: " << s.transpose() << "\n";
        std::cout << "Vt:\n" << Vt << "\n";
    }

    Eigen::Vector3f v2 = Vt.row(2).head<3>().stableNormalized();
    Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
    dir = v2.cross(v3);
    dir.stableNormalize();
    is_plane = true;

    Eigen::MatrixXf S_inv = Eigen::MatrixXf::Zero(4, 4);
    for (int i = 0; i < 4; ++i) {
        if (s[i] > 1e-6) {
            S_inv(i, i) = 1.0 / s[i];
        }
    }

    Eigen::MatrixXf A_inv = svd.matrixV() * S_inv * U.transpose();
    point = A_inv * Qb;
    return point;
}
std::optional<Eigen::Vector4f> SQEF::solve(double svd_threshold, bool debug) {
    if (A.rows() < 4) {
        int original_rows = A.rows();
        int original_cols = A.cols();
        A.conservativeResize(original_rows * 2, original_cols);
        A.bottomRows(original_rows) = A.topRows(original_rows);
        b.conservativeResize(original_rows * 2);
        b.tail(original_rows) = b.head(original_rows);
    }
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);

    Eigen::MatrixXf Q = qr.householderQ();//MxM
    Eigen::VectorXf Qb = (Q.transpose() * b).topRows(4);
    Eigen::MatrixXf R = qr.matrixQR().triangularView<Eigen::Upper>();//Mx4
    R.conservativeResize(4, 4);//R-->R_{top4}


    if (debug) {
        std::cout << "R:\n" << R << "\n";
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXf s = svd.singularValues();
    Eigen::MatrixXf U = svd.matrixU();
    Eigen::MatrixXf Vt = svd.matrixV().transpose();

    if (debug) {
        std::cout << "s: " << s.transpose() << "\n";
        std::cout << "Vt:\n" << Vt << "\n";
    }

    for (int i = 1; i < s.size(); ++i) {
        if (s[i] < s[0] * svd_threshold) {
            if (i == 1) {
                Eigen::Vector3f v1 = Vt.row(1).head<3>().stableNormalized();
                Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
                dir = v1.cross(v3);
                dir.stableNormalize();
                is_plane = true;
                return std::nullopt;
            } else if (i == 2) {
                Eigen::Vector3f v2 = Vt.row(2).head<3>().stableNormalized();
                Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();
                dir = v2.cross(v3);
                dir.stableNormalize();
                is_plane = true;
                break;
            } else if (i == 3) {
                dir = Vt.row(3).head<3>().stableNormalized();
                is_line = true;
                break;
            }
        }
    }

    Eigen::MatrixXf S_inv = Eigen::MatrixXf::Zero(4, 4);
    for (int i = 0; i < 4; ++i) {
        if (s[i] > 1e-6) {
            S_inv(i, i) = 1.0 / s[i];
        }
    }

    Eigen::MatrixXf A_inv = svd.matrixV() * S_inv * U.transpose();
    point = A_inv * Qb;
    return point;
}

//================================

void sqef_clear_type_state(SQEF& sqef)
{
    sqef.is_plane = false;
    sqef.is_line = false;
    sqef.dir = Eigen::Vector3f::Zero();

    // Python sets sqef.point = None. Since SQEF::point intentionally remains
    // Eigen::Vector4f, use NaN to catch accidental use before a new solve.
    sqef.point.setConstant(std::numeric_limits<float>::quiet_NaN());
}

std::tuple<Eigen::Vector4f, Eigen::MatrixXf>
sqef_factor_once(SQEF& sqef)
{
    MatF A = sqef.A;
    Eigen::VectorXf b = sqef.b;

    if (A.rows() < 4) {
        const Eigen::Index original_rows = A.rows();
        const Eigen::Index original_cols = A.cols();

        A.conservativeResize(original_rows * 2, original_cols);
        A.bottomRows(original_rows) = A.topRows(original_rows);

        b.conservativeResize(original_rows * 2);
        b.tail(original_rows) = b.head(original_rows);
    }

    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);

    Eigen::MatrixXf Q = qr.householderQ();
    Eigen::VectorXf Qb = (Q.transpose() * b).topRows(4);

    Eigen::MatrixXf R = qr.matrixQR().triangularView<Eigen::Upper>();
    R.conservativeResize(4, 4);

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(
        R,
        Eigen::ComputeFullU | Eigen::ComputeFullV
    );

    const Eigen::VectorXf s = svd.singularValues();
    const Eigen::MatrixXf U = svd.matrixU();
    const Eigen::MatrixXf V = svd.matrixV();
    const Eigen::MatrixXf Vt = V.transpose();

    Eigen::MatrixXf S_inv = Eigen::MatrixXf::Zero(4, 4);
    for (int i = 0; i < 4; ++i) {
        if (s(i) > 1e-6f) {
            S_inv(i, i) = 1.0f / s(i);
        }
    }

    Eigen::Vector4f point = V * S_inv * U.transpose() * Qb;
    return std::make_tuple(point, Vt);
}

float solve_x3(
    const MatF& A,
    const Eigen::VectorXf& b,
    const Eigen::Vector3f& x_known)
{
    if (A.cols() != 4) {
        throw std::invalid_argument("solve_x3: A must have 4 columns.");
    }
    if (A.rows() != b.size()) {
        throw std::invalid_argument("solve_x3: A.rows() must equal b.size().");
    }

    const Eigen::MatrixXf A_known = A.leftCols(3);
    const Eigen::VectorXf A_unknown = A.col(3);

    const Eigen::VectorXf b_adj = b - A_known * x_known;

    const float numerator = A_unknown.dot(b_adj);
    const float denominator = A_unknown.dot(A_unknown);

    if (std::abs(denominator) < 1e-12f) {
        throw std::runtime_error("solve_x3: A[:, 3] is zero.");
    }

    return numerator / denominator;
}

DualPoint solve_result_from_current_type(
    const SQEF& sqef,
    const Eigen::Vector3f& min_corner,
    const Eigen::Vector3f& max_corner,
    float corner_factor,
    float edge_factor,
    float plane_factor)
{
    const Eigen::Vector4f v = sqef.point;
    const Eigen::Vector3f pos = v.head<3>();

    const bool point_inside =
        is_inside(pos, min_corner, max_corner, corner_factor);

    if (!sqef.is_line && !sqef.is_plane) {
        if (point_inside) {
            const float radius = v(3);
            return DualPoint(
                pos,
                DualPointType::CORNER,
                Eigen::Vector3f::Zero(),
                true,
                radius
            );
        }
        return DualPoint::none(DualPointType::CORNER, Eigen::Vector3f::Zero(), false);
    }

    if (sqef.is_line) {
        auto [intersects, enter_point, exit_point] =
            line_aabb_intersection(
                pos,
                sqef.dir,
                min_corner,
                max_corner,
                edge_factor
            );

        if (intersects) {
            const Eigen::Vector3f best_v = (enter_point + exit_point) * 0.5f;
            const float radius = solve_x3(sqef.A, sqef.b, best_v);

            return DualPoint(
                best_v,
                DualPointType::EDGE,
                sqef.dir,
                true,
                radius
            );
        }

        return DualPoint::none(DualPointType::EDGE, sqef.dir, false);
    }

    if (sqef.is_plane) {
        MatF3 unique_intersections =
            plane_aabb_intersection(
                pos,
                sqef.dir,
                min_corner,
                max_corner,
                plane_factor
            );

        if (unique_intersections.rows() > 0) {
            const Eigen::Vector3f best_v =
                unique_intersections.colwise().mean().transpose();
            const float radius = solve_x3(sqef.A, sqef.b, best_v);

            return DualPoint(
                best_v,
                DualPointType::PLANE,
                sqef.dir,
                true,
                radius
            );
        }

        return DualPoint::none(DualPointType::PLANE, sqef.dir, false);
    }

    return DualPoint::none(DualPointType::PLANE, sqef.dir, false);
}

DualPoint get_neighbor_solution_of_current_type(
    const SQEF& sqef,
    const Eigen::Vector3f& center,
    const AABBBoxList& neighbor_boxes)
{
    const Eigen::Vector4f v = sqef.point;
    const Eigen::Vector3f pos = v.head<3>();

    if (!sqef.is_line && !sqef.is_plane) {
        for (const auto& box : neighbor_boxes) {
            const Eigen::Vector3f& adj_min_corner = box.first;
            const Eigen::Vector3f& adj_max_corner = box.second;

            if (is_inside(pos, adj_min_corner, adj_max_corner) && sqef.point(3) > 0.0f) {
                return DualPoint(
                    pos,
                    DualPointType::CORNER,
                    sqef.dir,
                    false,
                    sqef.point(3)
                );
            }
        }

        return DualPoint::none(DualPointType::CORNER, sqef.dir, false);
    }

    if (sqef.is_line) {
        float min_dis = std::numeric_limits<float>::infinity();
        std::optional<DualPoint> result = std::nullopt;

        for (const auto& box : neighbor_boxes) {
            const Eigen::Vector3f& adj_min_corner = box.first;
            const Eigen::Vector3f& adj_max_corner = box.second;

            auto [intersects, enter_point, exit_point] =
                line_aabb_intersection(
                    pos,
                    sqef.dir,
                    adj_min_corner,
                    adj_max_corner
                );

            if (intersects) {
                const Eigen::Vector3f best_v = (enter_point + exit_point) * 0.5f;
                const float radius = solve_x3(sqef.A, sqef.b, best_v);
                const float dis = (best_v - center).norm();

                if (radius > 0.0f && dis < min_dis) {
                    result = DualPoint(
                        best_v,
                        DualPointType::EDGE,
                        sqef.dir,
                        false,
                        radius
                    );
                    min_dis = dis;
                }
            }
        }

        if (result.has_value()) {
            return result.value();
        }
        return DualPoint::none(DualPointType::EDGE, sqef.dir, false);
    }

    // sqef.is_plane == true
    float min_dis = std::numeric_limits<float>::infinity();
    std::optional<DualPoint> result = std::nullopt;

    for (const auto& box : neighbor_boxes) {
        const Eigen::Vector3f& adj_min_corner = box.first;
        const Eigen::Vector3f& adj_max_corner = box.second;

        MatF3 unique_intersections =
            plane_aabb_intersection(
                pos,
                sqef.dir,
                adj_min_corner,
                adj_max_corner
            );

        if (unique_intersections.rows() > 0) {
            const Eigen::Vector3f best_v =
                unique_intersections.colwise().mean().transpose();
            const float radius = solve_x3(sqef.A, sqef.b, best_v);
            const float dis = (best_v - center).norm();

            if (radius > 0.0f && dis < min_dis) {
                result = DualPoint(
                    best_v,
                    DualPointType::PLANE,
                    sqef.dir,
                    false,
                    radius
                );
                min_dis = dis;
            }
        }
    }

    if (result.has_value()) {
        return result.value();
    }
    return DualPoint::none(DualPointType::PLANE, sqef.dir, false);
}

void force_line(SQEF& sqef)
{
    Eigen::Vector4f point;
    Eigen::MatrixXf Vt;
    std::tie(point, Vt) = sqef_factor_once(sqef);

    sqef_clear_type_state(sqef);

    sqef.point = point;
    sqef.is_line = true;
    sqef.is_plane = false;
    sqef.dir = Vt.row(3).head<3>().stableNormalized();
}

void force_plane(SQEF& sqef)
{
    Eigen::Vector4f point;
    Eigen::MatrixXf Vt;
    std::tie(point, Vt) = sqef_factor_once(sqef);

    sqef_clear_type_state(sqef);

    sqef.point = point;
    sqef.is_plane = true;
    sqef.is_line = false;

    const Eigen::Vector3f v2 = Vt.row(2).head<3>().stableNormalized();
    const Eigen::Vector3f v3 = Vt.row(3).head<3>().stableNormalized();

    sqef.dir = v2.cross(v3);
    sqef.dir.stableNormalize();
}

DualPoint solve_point_pre_sqef_with_group(
    SQEF& sqef,
    const Eigen::Vector3f& min_corner,
    const Eigen::Vector3f& max_corner,
    std::vector<std::vector<int>>& groups,
    bool is_cylinder,
    double svd_threshold,
    const AABBBoxList& one_ring_boxes,
    const AABBBoxList& two_ring_boxes,
    float corner_factor,
    float edge_factor,
    float plane_factor)
{
    const Eigen::Vector3f center = (min_corner + max_corner) * 0.5f;

    sqef_clear_type_state(sqef);

    std::optional<Eigen::Vector4f> v =
        sqef.solve_with_group(groups, is_cylinder, svd_threshold);

    if (!v.has_value()) {
        return DualPoint::none(DualPointType::PLANE, Eigen::Vector3f::Zero(), false);
    }

    DualPoint result = solve_result_from_current_type(
        sqef,
        min_corner,
        max_corner,
        corner_factor,
        edge_factor,
        plane_factor
    );

    if (result.m_point.has_value()) {
        return result;
    }

    result = get_neighbor_solution_of_current_type(sqef, center, one_ring_boxes);
    if (result.m_point.has_value()) {
        return result;
    }

    // Python intentionally skips two-ring fallback here.

    // corner -> line -> plane
    if (!sqef.is_line && !sqef.is_plane) {
        force_line(sqef);

        result = solve_result_from_current_type(
            sqef,
            min_corner,
            max_corner,
            corner_factor,
            edge_factor,
            plane_factor
        );
        if (result.m_point.has_value()) {
            return result;
        }

        result = get_neighbor_solution_of_current_type(sqef, center, one_ring_boxes);
        if (result.m_point.has_value()) {
            return result;
        }

        // Python intentionally skips two-ring fallback here as well.

        force_plane(sqef);

        result = solve_result_from_current_type(
            sqef,
            min_corner,
            max_corner,
            corner_factor,
            edge_factor,
            plane_factor
        );
        if (result.m_point.has_value()) {
            return result;
        }

        result = get_neighbor_solution_of_current_type(sqef, center, one_ring_boxes);
        if (result.m_point.has_value()) {
            return result;
        }

        return get_neighbor_solution_of_current_type(sqef, center, two_ring_boxes);
    }

    // line -> plane
    if (sqef.is_line) {
        force_plane(sqef);

        result = solve_result_from_current_type(
            sqef,
            min_corner,
            max_corner,
            corner_factor,
            edge_factor,
            plane_factor
        );
        if (result.m_point.has_value()) {
            return result;
        }

        result = get_neighbor_solution_of_current_type(sqef, center, one_ring_boxes);
        if (result.m_point.has_value()) {
            return result;
        }

        return get_neighbor_solution_of_current_type(sqef, center, two_ring_boxes);
    }

    // plane
    return get_neighbor_solution_of_current_type(sqef, center, two_ring_boxes);
}
