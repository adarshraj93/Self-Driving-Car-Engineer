#include "tools.h"
#include <iostream>
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    VectorXd residual;

    //  * the estimation vector size should not be zero
    if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
        //  * the estimation vector size should equal ground truth vector size
        cout << "Invalid estimation or ground_truth data!" << endl;
        return rmse;
    }
    //accumulate squared residuals
    for (unsigned int i = 0; i < estimations.size(); i++) {

        residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        rmse += residual;
    }

    // Calculate Mean
    rmse = rmse / estimations.size();
    
    // Calculate the squared root
    rmse = rmse.array().sqrt();

    // Return the RMSE value
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
    MatrixXd Hj(3, 4);

    Hj << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;
    //recover state parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    //pre-compute a set of terms to avoid repeated calculation
    double c1 = px * px + py * py;
    double c2 = sqrt(c1);
    double c3 = (c1 * c2);

    //check division by zero
    if (fabs(c1) < 0.0001) {
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        return Hj;
    }
    else {
        //compute the Jacobian matrix
        Hj << (px / c2), (py / c2), 0, 0,
            -(py / c1), (px / c1), 0, 0,
            py* (vx * py - vy * px) / c3, px* (px * vy - py * vx) / c3, px / c2, py / c2;
    }
    return Hj;
}
