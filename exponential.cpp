/*
 * Author: Arpit Dwivedi
 *
 * This code integrates the LBFGS solver from Yixuan's LBFGSpp repository
 * (https://github.com/yixuan/LBFGSpp) with the automatic differentiation
 * capabilities of the Ceres Solver (http://ceres-solver.org/).
 *
 * By combining these two open-source libraries, this project enables LBFGS
 * optimization with automatic differentiation for user-defined functions.
 *
 * If you use this code in your own work, please credit:
 *   Arpit Dwivedi
 * and acknowledge the use of both LBFGSpp and Ceres Solver open-source repositories.
 *
 * - LBFGSpp is used for efficient LBFGS optimization.
 * - Ceres Solver provides automatic differentiation via its Jet type.
 * - This integration allows you to optimize functions with gradients computed automatically.
 */


#include <iostream>
#include <limits>
#include <Eigen/Core>
#include <ceres/jet.h>
#include <LBFGSB.h>



using Scalar = double;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using ceres::Jet;

// Fit the model: y = exp(m*x + c)
template<typename T>
T exp_model(const T& x, const T& m, const T& c) {
    return ceres::exp(m * x + c);
}

class ExpFitAutoDiff {
private:
    const std::vector<double> x_data;
    const std::vector<double> y_data;

public:
    ExpFitAutoDiff(const std::vector<double>& x, const std::vector<double>& y)
        : x_data(x), y_data(y) {}

    Scalar operator()(const Vector& params, Vector& grad) {
        using JetT = Jet<Scalar, 2>;  // two parameters: m and c

        JetT m(params[0], Eigen::Vector2d(1.0, 0.0));  // ∂/∂m = 1, ∂/∂c = 0
        JetT c(params[1], Eigen::Vector2d(0.0, 1.0));  // ∂/∂m = 0, ∂/∂c = 1

        JetT loss = JetT(0.0);

        for (size_t i = 0; i < x_data.size(); ++i) {
            JetT y_pred = exp_model(JetT(x_data[i]), m, c);
            JetT residual = y_pred - y_data[i];
            loss += residual * residual;
        }

        grad[0] = loss.v[0];  // d(loss)/dm
        grad[1] = loss.v[1];  // d(loss)/dc

        return loss.a; // function value
    }
};

int main() {
    // Provided data
    const int kNumObservations = 67;
    const double data[] = {
      0.000000e+00, 1.133898e+00,
      7.500000e-02, 1.334902e+00,
      1.500000e-01, 1.213546e+00,
      2.250000e-01, 1.252016e+00,
      3.000000e-01, 1.392265e+00,
      3.750000e-01, 1.314458e+00,
      4.500000e-01, 1.472541e+00,
      5.250000e-01, 1.536218e+00,
      6.000000e-01, 1.355679e+00,
      6.750000e-01, 1.463566e+00,
      7.500000e-01, 1.490201e+00,
      8.250000e-01, 1.658699e+00,
      9.000000e-01, 1.067574e+00,
      9.750000e-01, 1.464629e+00,
      1.050000e+00, 1.402653e+00,
      1.125000e+00, 1.713141e+00,
      1.200000e+00, 1.527021e+00,
      1.275000e+00, 1.702632e+00,
      1.350000e+00, 1.423899e+00,
      1.425000e+00, 1.543078e+00,
      1.500000e+00, 1.664015e+00,
      1.575000e+00, 1.732484e+00,
      1.650000e+00, 1.543296e+00,
      1.725000e+00, 1.959523e+00,
      1.800000e+00, 1.685132e+00,
      1.875000e+00, 1.951791e+00,
      1.950000e+00, 2.095346e+00,
      2.025000e+00, 2.361460e+00,
      2.100000e+00, 2.169119e+00,
      2.175000e+00, 2.061745e+00,
      2.250000e+00, 2.178641e+00,
      2.325000e+00, 2.104346e+00,
      2.400000e+00, 2.584470e+00,
      2.475000e+00, 1.914158e+00,
      2.550000e+00, 2.368375e+00,
      2.625000e+00, 2.686125e+00,
      2.700000e+00, 2.712395e+00,
      2.775000e+00, 2.499511e+00,
      2.850000e+00, 2.558897e+00,
      2.925000e+00, 2.309154e+00,
      3.000000e+00, 2.869503e+00,
      3.075000e+00, 3.116645e+00,
      3.150000e+00, 3.094907e+00,
      3.225000e+00, 2.471759e+00,
      3.300000e+00, 3.017131e+00,
      3.375000e+00, 3.232381e+00,
      3.450000e+00, 2.944596e+00,
      3.525000e+00, 3.385343e+00,
      3.600000e+00, 3.199826e+00,
      3.675000e+00, 3.423039e+00,
      3.750000e+00, 3.621552e+00,
      3.825000e+00, 3.559255e+00,
      3.900000e+00, 3.530713e+00,
      3.975000e+00, 3.561766e+00,
      4.050000e+00, 3.544574e+00,
      4.125000e+00, 3.867945e+00,
      4.200000e+00, 4.049776e+00,
      4.275000e+00, 3.885601e+00,
      4.350000e+00, 4.110505e+00,
      4.425000e+00, 4.345320e+00,
      4.500000e+00, 4.161241e+00,
      4.575000e+00, 4.363407e+00,
      4.650000e+00, 4.161576e+00,
      4.725000e+00, 4.619728e+00,
      4.800000e+00, 4.737410e+00,
      4.875000e+00, 4.727863e+00,
      4.950000e+00, 4.669206e+00,
    };

    // Extract x and y values
    std::vector<double> x_vals, y_vals;
    for (int i = 0; i < kNumObservations; ++i) {
        x_vals.push_back(data[2 * i]);
        y_vals.push_back(data[2 * i + 1]);
    }

    ExpFitAutoDiff fun(x_vals, y_vals);

    // Initial guess for [m, c]
    Vector params(2);
    params << 0.0, 0.0;

    // Set bounds
    Vector lb(2), ub(2);
    lb << -std::numeric_limits<Scalar>::infinity(), -std::numeric_limits<Scalar>::infinity();
    ub <<  std::numeric_limits<Scalar>::infinity(),  std::numeric_limits<Scalar>::infinity();

    LBFGSpp::LBFGSBParam<Scalar> param;
    LBFGSpp::LBFGSBSolver<Scalar> solver(param);

    Scalar fx;
    int niter = solver.minimize(fun, params, fx, lb, ub);

    std::cout << "Solved in " << niter << " iterations\n";
    std::cout << "m = " << params[0] << ", c = " << params[1] << "\n";
    std::cout << "Final loss: " << fx << "\n";
    return 0;
}
