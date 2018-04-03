#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.512;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.743;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  Complete the initialization. See ukf.h for other member properties.
  */
  previous_timestamp_ = 0;
  is_initialized_ = false;
  
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2 * n_aug_ + 1);
  
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
}

UKF::~UKF() {}


// convert polar coordinates to cartesian
VectorXd polar2Cartesian(VectorXd polar){
  double px, py, vx, vy;
  double phi, rho;//, rho_n;
  VectorXd cartesian = VectorXd(5);
  
  rho = polar(0);
  phi = polar(1);
  //rho_n = polar(2);
  
  px = rho * cos(phi);
  py = rho * sin(phi);
  
  cartesian << px, py, 0, 0, 0;
  
  return cartesian;
}

// normalise the angle to be in range -pi to pi
double UKF::normAngle(const double theta){
    return atan2(sin(theta), cos(theta));
}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out){
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  double sq_3 = sqrt(3);
  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  //create augmented covariance matrix
  MatrixXd Q = MatrixXd(2, 2);
  Q << std_a_*std_a_, 0,
        0, std_yawdd_*std_yawdd_;
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q;
  
  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  MatrixXd sqA = sq_3*A;
  
  for(int i=0; i < n_aug_; i++){
      Xsig_aug.col(i+1) = x_aug + sqA.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - sqA.col(i);
  }

  //write result
  *Xsig_out = Xsig_aug;
}


void UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t){
  
  double dt_sq = delta_t * delta_t;
  double sinp, cosp, vk, phi, phi_n, n_acc, n_yaw;
  
  VectorXd xk = VectorXd(5);
  VectorXd A = VectorXd(5); // change vector
  VectorXd N = VectorXd(5); // noise vector
  //predict sigma points
  for(int i=0; i<(2*n_aug_ + 1); i++){
      vk = Xsig_aug(2, i);
      phi = Xsig_aug(3, i);
      phi_n = Xsig_aug(4, i);
      n_acc = Xsig_aug(5, i);
      n_yaw = Xsig_aug(6, i);
      sinp = sin(phi);
      cosp = cos(phi);
      
      xk << Xsig_aug(0, i), Xsig_aug(1, i), Xsig_aug(2, i), Xsig_aug(3, i), Xsig_aug(4, i);
      // avoid division by zero
      if(phi_n != 0){
        A(0) = (vk/phi_n)*(sin(phi + phi_n*delta_t) - sinp);
        A(1) = (vk/phi_n)*(-cos(phi + phi_n*delta_t) + cosp);
      }else{
        A(0) = vk * cosp * delta_t;
        A(1) = vk * sinp * delta_t;
      }
      
      A(2) = 0;
      A(3) = phi_n * delta_t;
      A(4) = 0;
      
      //noise vector
      N << dt_sq*cosp*n_acc/2, dt_sq*sinp*n_acc/2, delta_t*n_acc, dt_sq*n_yaw/2, delta_t*n_yaw;
      //update sigma point prediction
      Xsig_pred_.col(i) = xk + A + N;
  }
}


void UKF::PredictMeanAndCovariance(){
  int i;
  //set weights
  weights_(0) = (double)(lambda_/(lambda_ + n_aug_));
  for(i=1; i < 2 * n_aug_ + 1; i++){
      weights_(i) = (double)(0.5/(lambda_ + n_aug_));
  }
  //predict state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  
  //predict state covariance matrix
  P_.fill(0.0);
  for(i=0; i < 2 * n_aug_ + 1; i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = normAngle(x_diff(3));
    
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}


void UKF::PredictMeasurement(VectorXd &z_pred, MatrixXd &S, MatrixXd Zsig, MatrixXd R, int n_z){
  int i;
  //calculate mean predicted measurement
  z_pred = Zsig * weights_;
  //calculate innovation covariance matrix S
  MatrixXd Zdif = MatrixXd(n_z, 2 * n_aug_ + 1);
  MatrixXd Zdif_t = MatrixXd(2 * n_aug_ + 1, n_z);
  for(i=0; i<weights_.size(); i++){
      Zdif.col(i) = Zsig.col(i) - z_pred;
  }
  Zdif_t = Zdif.transpose();
  for(i=0; i<weights_.size(); i++){
      Zdif.col(i) = weights_(i) * Zdif.col(i);
  }
  S = (Zdif * Zdif_t) + R;
}


void UKF::UpdateState(MatrixXd S, MatrixXd Zsig, VectorXd z_pred, VectorXd z, int n_z, bool is_radar){
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i=0; i<weights_.size(); i++){
      VectorXd Xd = Xsig_pred_.col(i) - x_;
      if(is_radar)
        Xd(1) = normAngle(Xd(1));
      
      VectorXd Zd = Zsig.col(i) - z_pred;
      if(is_radar)
        Zd(1) = normAngle(Zd(1));
      
      Tc = Tc + weights_(i) * Xd * Zd.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //update state mean and covariance matrix
  //residual
  VectorXd z_diff = z - z_pred;
  //angle normalization, if RADAR
  if(is_radar)
    z_diff(1) = normAngle(z_diff(1));
  
  x_ = x_ + K*z_diff;
  P_ = P_ - K * S * K.transpose();
  
  nis_ = z_diff.transpose() * S * z_diff;
}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  //fixes simulator hanging error when restart simulation
  if(meas_package.timestamp_ < previous_timestamp_)
    is_initialized_ = false;
  
  if (!is_initialized_) {
    /**
      * Initialize the state x_ with the first measurement.
      * Convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "UKF: " << endl;
    //set the state with the initial location and zero velocity
    previous_timestamp_ = meas_package.timestamp_;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      x_ = polar2Cartesian(meas_package.raw_measurements_);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
       * Initialize state.
      */
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    // initialize predicted sigma points
    Xsig_pred_.fill(0.0);
    // initialize covariance matrix P_
    P_ = MatrixXd::Identity(n_x_, n_x_);
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  //compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;
  Prediction(dt);
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Any one of the following if-blocks will be executed at any instance
   */
  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
    // Laser updates
    UpdateLidar(meas_package);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    // Radar updates
    UpdateRadar(meas_package);
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
  cout << "NIS= " << nis_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_aug;
  AugmentedSigmaPoints(&Xsig_aug);
  SigmaPointPrediction(Xsig_aug, delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */  
  int n_z = 2;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  int i;
 
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;
  
  //transform sigma points into measurement space
  for(i=0; i<(2 * n_aug_ + 1); i++){
      double px = Xsig_pred_(0, i);
      double py = Xsig_pred_(1, i);
      
      Zsig.col(i) << px, py;
  }
  
  PredictMeasurement(z_pred, S, Zsig, R, n_z);
  UpdateState(S, Zsig, z_pred, meas_package.raw_measurements_, n_z, false);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

 double rho, phi, rho_d;
 int i;
 
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0, std_radrd_*std_radrd_;
        
  //transform sigma points into measurement space
  for(i=0; i<(2 * n_aug_ + 1); i++){
      double px = Xsig_pred_(0, i);
      double py = Xsig_pred_(1, i);
      double v = Xsig_pred_(2, i);
      double ph = Xsig_pred_(3, i);
      
      rho = sqrt(px*px + py*py);
      phi = atan2(py, px);
      rho_d = rho==0 ? 0 : (v*(px*cos(ph) + py*sin(ph)))/rho;
      
      Zsig.col(i) << rho, phi, rho_d;
  }
  PredictMeasurement(z_pred, S, Zsig, R, n_z);
  UpdateState(S, Zsig, z_pred, meas_package.raw_measurements_, 3, true);
}
