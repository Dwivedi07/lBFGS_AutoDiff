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

using namespace LBFGSpp;
using Scalar = double;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// Define the simple function using Ceres Jet
template<typename T>
T simple_quad(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x)
{
    return 0.5 * (10.0 - x[0]) * (10.0 - x[0]);
}

// Wrapper for LBFGS++ using autodiff
class SimpleQuadAutoDiff
{
private:
    int n;
public:
    SimpleQuadAutoDiff(int n_) : n(n_) {}

    Scalar operator()(const Vector& x, Vector& grad)
    {
        using ceres::Jet;
        typedef Jet<Scalar, Eigen::Dynamic> JetT;

        Eigen::Matrix<JetT, Eigen::Dynamic, 1> x_jet(n);
        for (int i = 0; i < n; ++i)
        {
            Eigen::VectorXd deriv = Eigen::VectorXd::Zero(n);
            deriv[i] = 1.0;
            x_jet[i] = JetT(x[i], deriv);  // Assign value and partial
        }

        JetT f_jet = simple_quad(x_jet);
        for (int i = 0; i < n; ++i)
            grad[i] = f_jet.v[i];

        return f_jet.a;
    }
};

int main()
{
    const int n = 1;
    SimpleQuadAutoDiff fun(n);

    LBFGSBParam<Scalar> param;
    param.epsilon = 1e-8;
    param.max_iterations = 100;
    LBFGSBSolver<Scalar> solver(param);

    // Initial guess
    Vector x(n);
    x[0] = 0.0;

    // Optional bounds
    Vector lb = Vector::Constant(n, -5.0);
    Vector ub = Vector::Constant(n, 15.0);

    Scalar fx;
    int niter = solver.minimize(fun, x, fx, lb, ub);

    std::cout << niter << " iterations\n";
    std::cout << "x = " << x[0] << "\n";
    std::cout << "f(x) = " << fx << "\n";
    std::cout << "grad = " << solver.final_grad()[0] << "\n";
    std::cout << "projected grad norm = " << solver.final_grad_norm() << std::endl;

    return 0;
}
