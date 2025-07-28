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

// Automatic differentiation with Ceres::Jet
template<typename T>
T rosenbrock_autodiff(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x)
{
    T fx = (x[0] - 1.0) * (x[0] - 1.0);
    for (int i = 1; i < x.size(); ++i)
    {
        T t = x[i] - x[i - 1] * x[i - 1];
        fx += 4.0 * t * t;
    }
    return fx;
}

// Wrapper for LBFGS++ to use autodiff
class RosenbrockAutoDiff
{
private:
    int n;
public:
    RosenbrockAutoDiff(int n_) : n(n_) {}

    Scalar operator()(const Vector& x, Vector& grad)
    {
        using ceres::Jet;
        typedef Jet<Scalar, Eigen::Dynamic> JetT;

        Eigen::Matrix<JetT, Eigen::Dynamic, 1> x_jet(n);
        for (int i = 0; i < n; ++i)
        {
            Eigen::VectorXd deriv = Eigen::VectorXd::Zero(n);
            deriv[i] = 1.0;
            x_jet[i] = JetT(x[i], deriv);  // set ith variable with derivative 1
        }

        JetT f_jet = rosenbrock_autodiff(x_jet);

        for (int i = 0; i < n; ++i)
            grad[i] = f_jet.v[i];  // extract gradient from Jet

        return f_jet.a; // function value
    }
};

int main()
{
    const int n = 25;
    LBFGSBParam<Scalar> param;
    LBFGSBSolver<Scalar> solver(param);
    RosenbrockAutoDiff fun(n);

    // Variable bounds
    Vector lb = Vector::Constant(n, 2.0);
    Vector ub = Vector::Constant(n, 4.0);
    lb[2] = -std::numeric_limits<Scalar>::infinity(); // unbounded
    ub[2] = std::numeric_limits<Scalar>::infinity();

    // Initial guess
    Vector x = Vector::Constant(n, 3.0);
    x[0] = x[1] = 2.0;
    x[5] = x[7] = 4.0;

    Scalar fx;
    int niter = solver.minimize(fun, x, fx, lb, ub);

    std::cout << niter << " iterations\n";
    std::cout << "x = \n" << x.transpose() << "\n";
    std::cout << "f(x) = " << fx << "\n";
    std::cout << "grad = \n" << solver.final_grad().transpose() << "\n";
    std::cout << "projected grad norm = " << solver.final_grad_norm() << std::endl;

    return 0;
}