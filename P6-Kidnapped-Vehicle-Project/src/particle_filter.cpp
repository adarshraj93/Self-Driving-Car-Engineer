/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
using std::string;
using std::vector;

/**
  * Define global Random number engine class that generates random pseudo numbers
  * Defined globally so we can access from all the funtion
 */
static default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  num_particles = 125;  // TODO: Set the number of particles

  // Define and set the standard deviations for x,y & theta
  double std_x, std_y, std_theta; //Define
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // Allocate memory to the vectors: 1.particle 2.weights
  particles.reserve(num_particles);
  weights.reserve(num_particles);

  // Generate Normal Gaussian Distribution functions for x,y & theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // Initialize the particles and sample from the normal distribution
  for (int i = 0; i < num_particles; ++i) {
      // Create an object with Particle structure for initalization
      Particle obj_p;
      obj_p.id = i;
      obj_p.x = dist_x(gen);
      obj_p.y = dist_y(gen);
      obj_p.theta = dist_theta(gen);
      obj_p.weight = 1.0;
      
      //Append the values into particles
      particles.push_back(obj_p);
      weights.push_back(obj_p.weight);
  }
  // Set initialisation Flag to true after intialisation
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Define and get the standard deviation for x,y & theta
  double std_x, std_y, std_theta; //Define
  std_x = std_pos[0];
  std_y = std_pos[1];
  std_theta = std_pos[2];

  // Generate random Gaussian noise  for x,y & theta with mean 0 and 
  // standard deviation as std_x, std_y, std_theta
  normal_distribution<double> noise_x(0, std_x);
  normal_distribution<double> noise_y(0, std_y);
  normal_distribution<double> noise_theta(0, std_theta);

  // Predict the vehicle location with velocity and certain heading after delta t
  for (int i = 0; i < num_particles; ++i) {
      if (fabs(yaw_rate) < 0.0001) {// when yaw rate is 0
          particles[i].x += velocity * delta_t * cos(particles[i].theta);
          particles[i].y += velocity * delta_t * sin(particles[i].theta);
      }
      else {
          // Calculate the term delta_theta (yaw change over time delta_t)
          double delta_theta = yaw_rate * delta_t;

          //Predict the final x,y & theta
          particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + delta_theta) - sin(particles[i].theta));
          particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + delta_theta));
          particles[i].theta += delta_theta;
      }
      // Add the noise to the prediction
      particles[i].x += noise_x(gen);
      particles[i].y += noise_y(gen);
      particles[i].theta += noise_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  //For loop to go through all the observations or sensor measurements
  for (int i = 0; i < observations.size(); ++i) {

      // Define the current observations 
      LandmarkObs cur_obs = observations[i];

      // Define & initialize mininum distance to max value
      double min_dist = numeric_limits<double>::max();

      // Define & initialize the landmark id
      int id = -1;

      //For loop to go through all the predictions
      for (int j = 0; j < predicted.size(); ++j) {
          
          // Define current prediction
          LandmarkObs cur_pred = predicted[j];

          // Calculate the distance b/n the current observation and the prediction
          double cur_dist;
          cur_dist = dist(cur_obs.x, cur_obs.y, cur_pred.x, cur_pred.y);

          //Check for the closest distance of each prediction
          // Update min_dist with cur_dist if smaller 
          //Save the prediction ID to the landmark id
          if (cur_dist < min_dist) {
              min_dist = cur_dist;
              id = cur_pred.id;
          }
      }
      // Assign the measured landmark id (this is the closest neighbour) to the observation id
      observations[i].id = id;
  }
 

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // Precalculate constants and variables to save computation time
  const double x_std = std_landmark[0];
  const double y_std = std_landmark[1];
  const double x_var = x_std * x_std;
  const double y_var = y_std * y_std;
  const double gauss_norm = 2 * M_PI * x_std * y_std;

  // Loop to read all the particles
  for (int i = 0; i < particles.size(); ++i) {
      // Particle parameters
      double x_p = particles[i].x;
      double y_p = particles[i].y;
      double theta_p = particles[i].theta;
      particles[i].weight = 1.0;

      // 1. Transform landmark observations from car to particle coordinate system

      // Define vector to store the transformed observation
      std::vector<LandmarkObs> obs_trans;
      obs_trans.reserve(observations.size()); //Reserve memory as size is known

      // Define a temporary variable to store transformation of ith observation
      LandmarkObs obs_tmp;

      // Loop to read all the observations
      for (int j = 0; j < observations.size(); ++j) {
          obs_tmp.id = observations[j].id;
          obs_tmp.x = observations[j].x * cos(theta_p) - observations[j].y * sin(theta_p) + x_p;
          obs_tmp.y = observations[j].x * sin(theta_p) + observations[j].y * cos(theta_p) + y_p;

          // Append the transformed obeservation coordinates
          obs_trans.push_back(obs_tmp);
      }

      // 2. Identify landmarks within the sensor_range from the particle
      // Define vector for the selected landmarks
      std::vector<LandmarkObs> predictions;

      // Loop through all the landmarks
      for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
          // Store landmark x,y & id in tmp variable
          obs_tmp.id = map_landmarks.landmark_list[k].id_i;
          obs_tmp.x = map_landmarks.landmark_list[k].x_f;
          obs_tmp.y = map_landmarks.landmark_list[k].y_f;

          //Append the landmarks within the sensor_range
          if (dist(obs_tmp.x, obs_tmp.y, x_p, y_p) <= sensor_range) {
              predictions.push_back(obs_tmp);
          }
      }

      // 3. Call dataAssociation function to find the closest landmark to the obeservation
      dataAssociation(predictions, obs_trans);

      // 4. Update weights of particles using Multi-Variante Gaussian Distribution

      //Loop through all transformed obeservations
      for (int j = 0; j < obs_trans.size(); ++j) {
          // Define variables for nearest landmark id, x & y
          int nl_id = obs_trans[j].id; // Defined and initiliazed 
          double nl_x, nl_y;

          //Set weight to 0 if no landmark is associated with observation
          if (nl_id == -1) {
              particles[i].weight = 0;
              break;
          }
          else {
              // Nearest landmark coordinates (nl_x,nl_y) comes from predictions 
              for (unsigned int m = 0; m < predictions.size(); ++m) {
                  if (predictions[m].id == nl_id) {
                      nl_x = predictions[m].x;
                      nl_y = predictions[m].y;
                  }
              }
              // Observed coordinates
              double ob_x = obs_trans[j].x;
              double ob_y = obs_trans[j].y;
              
              // Calculate importance weight with MultiVariant Gaussian
              double gauss_exp = exp(-(pow(nl_x - ob_x, 2) / (2 * pow(x_std, 2)) + (pow(nl_y - ob_y, 2) / (2 * pow(y_std, 2)))));
              double imp_w = gauss_exp / gauss_norm;

              // Update particle weight for the current observation
              particles[i].weight *= imp_w;
          }   
      }
      // Update the weights vector
      weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Define and allocate memory for storing resampled particles
  vector<Particle> resample_p;
  resample_p.reserve(num_particles);

  // Define the discrete distribution
  discrete_distribution<int> dist(weights.begin(), weights.end());

  // Loop trhough all particles
  for (int i = 0; i < num_particles; ++i) {
      resample_p.push_back(particles[dist(gen)]);
  }

  // Update particles with resampled particles
  particles = resample_p;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}