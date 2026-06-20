#ifndef PLANE_HPP
#define PLANE_HPP

#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iomanip>

class Plane {
public:
    Eigen::Vector3f normal;
    Eigen::Vector3f point;
    float distance;
    Plane(){
        
    }
    Plane(const Eigen::Vector3f& normal, const Eigen::Vector3f& point) 
        : normal(normal.normalized()), point(point) {
        distance = -normal.dot(point);
    }

    Plane(const Eigen::Vector3f& normal, float distance)
        : normal(normal.normalized()), distance(distance) {
        if (std::abs(normal[0]) > 1e-8) {
            point = Eigen::Vector3f(-distance/normal[0], 0, 0);
        } else if (std::abs(normal[1]) > 1e-8) {
            point = Eigen::Vector3f(0, -distance/normal[1], 0);
        } else {
            point = Eigen::Vector3f(0, 0, -distance/normal[2]);
        }
    }
    
    std::string toString() const {
        std::ostringstream ss;
        ss.precision(2);
        ss << std::fixed;
        
        bool first_term = true;
        if (std::abs(normal[0]) > 1e-8) {
            ss << normal[0] << "x";
            first_term = false;
        }
        if (std::abs(normal[1]) > 1e-8) {
            if (!first_term && normal[1] > 0) ss << "+";
            ss << normal[1] << "y";
            first_term = false;
        }
        if (std::abs(normal[2]) > 1e-8) {
            if (!first_term && normal[2] > 0) ss << "+";
            ss << normal[2] << "z";
            first_term = false;
        }
        if (std::abs(distance) > 1e-8) {
            if (!first_term && distance > 0) ss << "+";
            ss << distance;
        }
        ss << "=0";
        return ss.str();
    }
    
    float distanceToPoint(const Eigen::Vector3f& point) const {
        return std::abs(normal.dot(point) + distance) / normal.norm();
    }
};

#endif // PLANE_HPP