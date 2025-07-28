# LBFGS Autodiff Integration

**Author:** Arpit Dwivedi

## Overview

This repository demonstrates how to integrate the [LBFGSpp](https://github.com/yixuan/LBFGSpp) solver with the automatic differentiation capabilities of the [Ceres Solver](http://ceres-solver.org/). By combining these two open-source libraries, you can perform efficient LBFGS optimization with gradients computed automatically for user-defined functions.

## Features

- **LBFGSpp:** Efficient LBFGS optimization for unconstrained and bound-constrained problems.
- **Ceres Solver:** Automatic differentiation using the Jet type.
- **Examples:** Includes quadratic and exponential fitting examples using automatic differentiation and LBFGS optimization.

## Usage

1. **Clone the repository:**
   ```sh
   git clone https://github.com/<your-username>/lbfgs_autodiff.git
   cd lbfgs_autodiff
   ```

2. **Install dependencies:**
   - [Eigen](https://eigen.tuxfamily.org/)
   - [LBFGSpp](https://github.com/yixuan/LBFGSpp)
   - [Ceres Solver](http://ceres-solver.org/)

3. **Build the examples:**
   Use your preferred C++ build system (e.g., CMake or g++ directly).

   Example with g++:
   ```sh
   g++ quadratic.cpp -o quadratic -I/path/to/eigen -I/path/to/lbfgspp/include -I/path/to/ceres/include -L/path/to/ceres/lib -lceres
   g++ exponential.cpp -o exponential -I/path/to/eigen -I/path/to/lbfgspp/include -I/path/to/ceres/include -L/path/to/ceres/lib -lceres
   ```

4. **Run the examples:**
   ```sh
   ./quadratic
   ./exponential
   ```

## Citation

If you use this code in your work, please credit:

**Arpit Dwivedi**

and acknowledge the use of both LBFGSpp and Ceres Solver open-source repositories.

## License

See the respective licenses for LBFGSpp and Ceres Solver. This integration is provided under an open-source license.

## Acknowledgements

- [LBFGSpp by Yixuan Qiu](https://github.com/yixuan/LBFGSpp)
- [Ceres Solver](http://ceres-solver.org/)

## Contact

For questions or suggestions, please open an issue or contact Arpit Dwivedi.
