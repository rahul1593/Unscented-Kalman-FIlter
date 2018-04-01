#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /*
   * Calculate the RMSE here.
   */
  VectorXd rmse(4);
	rmse << 0, 0, 0, 0;
	//input vector length
	int inpv_ln = estimations.size();
	//check the size of input vectors and return zero RMSE if invalid input
	if( inpv_ln == 0 || inpv_ln != ground_truth.size()){
		return rmse;
	}
	//accumulate the difference squares
	for( int i=0; i < inpv_ln; i++){
		VectorXd df = estimations[i] - ground_truth[i];
		// element by element multiplication
		df = df.array() * df.array();
		rmse += df;
	}
	// mean
	rmse = rmse/inpv_ln;
	//square root
	rmse = rmse.array().sqrt();

	return rmse;
}
