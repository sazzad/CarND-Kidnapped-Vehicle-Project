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
#include "Eigen/Dense"


#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 500;  // TODO: Set the number of particles
  std::default_random_engine gen;


  // This line creates a normal (Gaussian) distribution for x
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i=0; i<num_particles; ++i){
    Particle p(i, dist_x(gen), dist_y(gen), dist_theta(gen));
    particles.push_back(p);
  }
  is_initialized  = true;
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

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  for(int i=0;i<num_particles;++i){
    Particle& p = particles[i];

    double theta_f = p.theta + yaw_rate * delta_t;
    double x_f = p.x + (velocity/yaw_rate) * ( sin(theta_f) - sin(p.theta));
    double y_f = p.y + (velocity/yaw_rate) * ( cos(p.theta) - cos(theta_f));

    p.x = x_f + dist_x(gen);
    p.y = y_f + dist_y(gen);
    p.theta = theta_f + dist_theta(gen);
    //std::cout<<"p.x: "<< p.x << " p.y: "<< p.y<< " theta: "<< p.theta<<" weight: "<< p.weight<<std::endl;

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

}

Eigen::Vector2d FindNN(const Map& map_landmarks, Eigen::Vector2d& ob, int* id){

  double min_dist = std::numeric_limits<double>::max();
  Eigen::Vector2d nearest_landmark;

  const auto& landmarks = map_landmarks.landmark_list;

  for(int i=0;i<landmarks.size();++i){
    Eigen::Vector2d v(landmarks[i].x_f, landmarks[i].y_f);
    double dist  = (ob - v).norm();
    if(dist < min_dist){
      min_dist = dist;
      nearest_landmark = v;
      *id= landmarks[i].id_i;
    }
  }
  return nearest_landmark;

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

  // car frame to global frame: car frame : x forward, y left: map frame; y forward, x right.

  std::function<double(double)> sq = [](double x){return x*x; };
  Eigen::Matrix2d P;
  P << sq(std_landmark[0]), 0,
        0, sq(std_landmark[1]);

  std::function<double(const Eigen::Vector2d&)> Gaussian = [&](const Eigen::Vector2d& x){
    return (1./(2* M_PI * std_landmark[0]*std_landmark[1])) * exp(-0.5 * x.transpose() * P.inverse() * x);
  };

  weights.clear();

  double total_w = 0;
  for(int i=0;i<num_particles;++i){
    Particle& p = particles[i];

    // clear previous associations.
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();

    for(const LandmarkObs& ob : observations){
      // 1. check if the observation is below sensor range or not.
      double range = sqrt(sq(ob.x) + sq(ob.y));
      //std::cout<<"range: "<< range << std::endl;
      if(range>sensor_range){
        continue;
      }

      // 2. convert observation from car frame to map frame.
      Eigen::Matrix3d M;
      M<<cos(p.theta), -sin(p.theta), p.x, 
        sin(p.theta), cos(p.theta), p.y, 
        0.0, 0.0, 1.0;

      Eigen::Vector3d C_p_observation;
      C_p_observation << ob.x, ob.y, 1.0;
      Eigen::Vector3d ob_map = M * C_p_observation;

      Eigen::Vector2d G_p_observation(ob_map(0), ob_map(1));
      int landmark_id=0;
      Eigen::Vector2d G_p_landmark = FindNN(map_landmarks, G_p_observation, &landmark_id);

      p.associations.push_back(landmark_id);
      p.sense_x.push_back(G_p_observation(0));
      p.sense_y.push_back(G_p_observation(1));


      double posterior = Gaussian(G_p_observation - G_p_landmark);
      //std::cout<<"Posterior: "<< posterior<< std::endl;
      p.weight*=posterior;
    }

    total_w+=p.weight;

  }
  // normalize:
  
  for(int i=0;i<num_particles;++i){
    Particle& p = particles[i];
    p.weight/=total_w;
    weights.push_back(p.weight);
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::default_random_engine gen;
  std::uniform_int_distribution<> dis(0, num_particles-1);

  auto iter = std::max_element(weights.begin(), weights.end());
  std::cout<<"max weight: "<< *iter<<"\n";

  std::vector<Particle> resampled_particles;
  int index = dis(gen);
  double beta = 0;
  for(int i=0;i<num_particles;++i){
    std::uniform_real_distribution<> dis2(0,  2 * (*iter));
    beta += dis2(gen);
    while(beta > weights[index]){
      beta -= weights[index];
      index = (index+1)%num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }
  particles.clear();
  particles = resampled_particles;

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