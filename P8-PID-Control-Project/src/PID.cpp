#include "PID.h"
#include <vector>
#include <limits>
#include <iostream>

using namespace std;

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */

	Kp = Kp_;
	Ki = Ki_;
	Kd = Kd_;
	p_error = 0;
	i_error = 0;
	d_error = 0;
	prev_cte = 0;
}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */

	// Proportional Error
	p_error = cte;
	// Derivative Error
	d_error = cte - prev_cte;
	prev_cte = cte; // Update the previous error to current error for the next iteration
	// Integral Error
	i_error += cte;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
	
	double total_error = Kp * p_error + Ki * i_error + Kd * d_error;
  
	return total_error;  // TODO: Add your total error calc here!
}