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

// Powell function with Ceres Jet autodiff
template<typename T>
T powell_autodiff(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x)
{
    T term1 = x[0] + 10.0 * x[1];
    T term2 = x[2] - x[3];
    T term3 = x[1] - 2.0 * x[2];
    T term4 = x[0] - x[3];

    return term1 * term1 + 5.0 * term2 * term2 + pow(term3, 4) + 10.0 * pow(term4, 4);
}

// Objective function wrapper for LBFGS++
class PowellAutoDiff
{
private:
    int n;
public:
    PowellAutoDiff(int n_) : n(n_) {}

    Scalar operator()(const Vector& x, Vector& grad)
    {
        using ceres::Jet;
        typedef Jet<Scalar, Eigen::Dynamic> JetT;

        Eigen::Matrix<JetT, Eigen::Dynamic, 1> x_jet(n);
        for (int i = 0; i < n; ++i)
        {
            Eigen::VectorXd deriv = Eigen::VectorXd::Zero(n);
            deriv[i] = 1.0;
            x_jet[i] = JetT(x[i], deriv);  // Assign value and gradient
        }

        JetT f_jet = powell_autodiff(x_jet);

        for (int i = 0; i < n; ++i)
            grad[i] = f_jet.v[i];  // Gradient from Jet

        return f_jet.a;  // Function value
    }
};

int main()
{
    const int n = 4;
    PowellAutoDiff fun(n);

    LBFGSBParam<Scalar> param;
    param.epsilon = 1e-8;
    param.max_iterations = 500;
    LBFGSBSolver<Scalar> solver(param);

    // Initial guess
    Vector x(n);
    x << 3.0, -1.0, 0.0, 1.0;

    // Variable bounds
    Vector lb = Vector::Constant(n, -2.0);
    Vector ub = Vector::Constant(n, 2.0);
    lb[2] = -std::numeric_limits<Scalar>::infinity();  // Unbounded
    ub[2] = std::numeric_limits<Scalar>::infinity();   // Unbounded

    Scalar fx;
    int niter = solver.minimize(fun, x, fx, lb, ub);

    std::cout << niter << " iterations\n";
    std::cout << "x = \n" << x.transpose() << "\n";
    std::cout << "f(x) = " << fx << "\n";
    std::cout << "grad = \n" << solver.final_grad().transpose() << "\n";
    std::cout << "projected grad norm = " << solver.final_grad_norm() << std::endl;

    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
