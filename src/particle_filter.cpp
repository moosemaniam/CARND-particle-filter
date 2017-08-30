/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define NUM_PARTICLES 100
#define YAW_RATE_MIN 0.0001
using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	//  Set the number of particles.
  num_particles = NUM_PARTICLES;
  is_initialized = false;
  normal_distribution<double>n_x_init(0,std[0]);
  normal_distribution<double>n_y_init(0,std[1]);
  normal_distribution<double>n_theta_init(0,std[2]);

  for (int i=0;i<num_particles;i++){
    Particle p;
    p.id=i;

  //Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
    p.x= x + n_x_init(gen);
    p.y= y + n_y_init(gen);
    p.theta = theta + n_theta_init(gen);

    p.weight = 1.0f;
    /* Store the particles */
    particles.push_back(p);
  }

  is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double>n_x(0,std_pos[0]);
  normal_distribution<double>n_y(0,std_pos[1]);
  normal_distribution<double>n_theta(0,std_pos[2]);


  for(int i=0;i< num_particles;i++){
    //Predict new state

    double p_theta = particles[i].theta;
    if(fabs(yaw_rate) < YAW_RATE_MIN)
    {
      /* Change in yaw rate is small */
      particles[i].x += velocity*delta_t*cos(p_theta);
      particles[i].y += velocity*delta_t*sin(p_theta);
    }
    else
    {
      particles[i].x +=  velocity/yaw_rate * (sin(p_theta+yaw_rate*delta_t) - sin(p_theta));
      particles[i].y +=  velocity/yaw_rate * (cos(p_theta)-cos(p_theta+yaw_rate*delta_t));
      particles[i].theta += yaw_rate*delta_t;
    }

    /* Add gaussian noise to the prediction */
    particles[i].x += n_x(gen);
    particles[i].y += n_y(gen);
    particles[i].theta += n_theta(gen);
  }

}

double calculate_distance(double x1, double y1,double x2,double y2)
{
  return(sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)));
}
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  //

  for(int i =0;i< observations.size();i++){
    double minval;
    double map_index = 0;
    double x_ob=observations[i].x;
    double y_ob=observations[i].y;
    double dist;
    double x_p = predicted[0].x;
    double y_p = predicted[0].y;

    dist = calculate_distance(x_p,y_p,x_ob,y_ob);
    minval = dist;
    map_index = predicted[0].id;

    for(int l=1;l<predicted.size();l++){

      x_p = predicted[l].x;
      y_p = predicted[l].y;

      dist = calculate_distance(x_p,y_p,x_ob,y_ob);

      if(dist < minval){
        minval = dist;
        map_index = predicted[l].id;
      }
    }
    observations[i].id = map_index;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  //
  //

   double dist;
  /* particle loop */
  for(int i=0;i < num_particles;i++){


    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    vector<LandmarkObs> predictions;

    /* landmark loop */
    for(int l=0; l < map_landmarks.landmark_list.size();l++)
    {

      double lmx = map_landmarks.landmark_list[l].x_f;
      double lmy = map_landmarks.landmark_list[l].y_f;
      int lmid = map_landmarks.landmark_list[l].id_i;

      /* Use prediction only if the landmark is within sensor range of the particle
       **/
      dist = calculate_distance(x,y,lmx,lmy);

      if(dist <= sensor_range)
      {
        /* add prediction */
        predictions.push_back(LandmarkObs{lmid,lmx,lmy,});
      }
    }//landmark loop

    /* Convert observations from vehicle to map co-ordinate system */
    vector<LandmarkObs> map_coordinates_obs;

    for(int l =0;l < observations.size();l++){
      double map_x = cos(theta) *observations[l].x - sin(theta)*observations[l].y + x;
      double map_y = sin(theta) *observations[l].x + cos(theta)*observations[l].y + y;
      map_coordinates_obs.push_back(LandmarkObs{observations[l].id,map_x,map_y});
    }

    /* call data association for the predictions and transformed(map) coordinates of this particle */
    dataAssociation(predictions,map_coordinates_obs);

    /* Re-calculate weight of the particle */
    particles[i].weight = 1.0;

    for(int l=0;l<map_coordinates_obs.size();l++){

      int id;
      double obs_x,obs_y;
      double predicted_x,predicted_y;

      obs_x = map_coordinates_obs[l].x;
      obs_y = map_coordinates_obs[l].y;
      id = map_coordinates_obs[l].id;

      /* co-ordinates of the predictions associated with present observation */

      for(int ii=0;ii<predictions.size();ii++){
        if(predictions[ii].id ==  id){
          predicted_x = predictions[ii].x;
          predicted_y = predictions[ii].y;
        }
      }

      /* Weight calculations */
      double std_x =  std_landmark[0];
      double std_y =  std_landmark[1];

      double exponent = -((pow(predicted_x-obs_x,2)/(2*std_x*std_x))+(pow(predicted_y-obs_y,2)/(2*std_y*std_y)));

      double weight_m = (1/(2*M_PI*std_x*std_y))*exp(exponent);

      particles[i].weight *= weight_m;
    }
  }
}

void ParticleFilter::resample()
{
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  //
  vector<Particle> resampled_particles;
  double beta=0.0f;
  /* Generate a random index */
  uniform_int_distribution<int> random_int(0,num_particles-1);
  int index = random_int(gen);

  vector<double> weights;
  for(int i=0;i<num_particles;i++){
    weights.push_back(particles[i].weight);
  }

  /* Create a discrete random number generator based on the weights of */
  /* the particles */
  double max = *max_element(weights.begin(),weights.end());
  std::uniform_real_distribution<double> float_random(0.0,max);

  //from sebastian's tutorial
  for(int i=0;i<num_particles;i++){
    double random_val = float_random(gen);
    beta +=  random_val * 2.0;
#if 1
    while(beta > weights[index]){
      beta -= weights[index];
      index = (index + 1)%num_particles;
    }
#endif
    resampled_particles.push_back(particles[index]);
  }
  /* Update resampled particles */
  particles=resampled_particles;
  for(int i=0;i<num_particles;i++){
    printf("After update weights[%d] %f\n",i,particles[i].weight);
  }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
