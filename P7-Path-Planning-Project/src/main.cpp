#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors. These are the points on the map
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0. This is the length of the track
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  //Retrieve x, y, s, d_x, d_y of the waypoints from the map and assign them to the respective vectors defined above
  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // Start in lane 1 which is the middle lane (-1:No Lane, 0:leftmost_lane, 1:middle_lane, 2:rightmost_lane)
  int lane = 1;

  // Define the reference velocity of the car in MPH (set as 0, max:49.5 MPH)
  //double ref_vel = 49.5; //mph
  double ref_vel = 0.0; //mph to remove the cold start issue and we have set the logic to increment it at line 140

  h.onMessage([&ref_vel,&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy,&lane]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          // of the road. Target Car is used as the identification for other cars
          //auto sensor_fusion = j[1]["sensor_fusion"];
          vector<vector<double>> sensor_fusion = j[1]["sensor_fusion"];

          // START CODE FOR PATH PLANNER
          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds. Model car is used as identification for our car
           */

          /* Code for Project written here
            1. Check for target vehicles (other vehicle) in lanes
            2. Predict where the target vehicle will be in future
            3. Set the flag where target vehicle is present (this tells which lane is open for lane change)
            4. Develop the trajecorty of lane change procedure
          */
          
          // Fetch number of points from previous path. The simulator gives the value
          int prev_size = previous_path_x.size();

          // If previous path exists, then set the starting point as the end point of the previous path
          if (prev_size > 0)
          {
              car_s = end_path_s;
          }

          // Variable to store target (other) car lane value.
          int target_car_lane;

          // Flags to indicate the presence of a car in a particular lane
          //bool too_close = false;
          bool target_car_front = false;
          bool target_car_left = false;
          bool target_car_right = false;

          // 1. Prediction Step 1 - Use Sensor Fusion data to identify target cars in different lanes and
          // and the respective velocities
          for (int i = 0; i < sensor_fusion.size(); i++)
          {
              // From README information, fetch the respective 'd' value of the target cars from sensor fusion data
              float d_target = sensor_fusion[i][6];

              // 1. Prediction: Identify the target car's lane lines and set target_car_lane,
              // (-1:No Lane, 0:leftmost_lane, 1:middle_lane, 2:rightmost_lane):
              // 1. If 0 < d_target < 4 - Left Lane
              // 2. If 4 < d_target < 8 - Middle Lane
              // 3. If 8 < d_target < 12 - Right Lane
              if (d_target > 0 && d_target < 4)
              {
                  target_car_lane = 0;  //Left Lane
              }
              else if (d_target > 4 && d_target < 8)
              {
                  target_car_lane = 1;  //Middle Lane
              }
              else if (d_target > 8 && d_target < 12)
              {
                  target_car_lane = 2;  //Right Lane
              }
              else
              {
                  target_car_lane = -1;     // No Lane
              }

              // From README information, fetch the speed components of the target cars from sensor fusion data
              double vx = sensor_fusion[i][3];
              double vy = sensor_fusion[i][4];
              // Calculate the speeed magnitude from the components. This is speed to be used with teh frenet co-ordinates
              double target_car_speed = sqrt(vx * vx + vy * vy);
              // From README information, fetch the respective 's' value of the target cars from sensor fusion data
              double target_car_s = sensor_fusion[i][5];

              /* 2. Prediction Step 2: Determine target car's 's' position at the end of current cycle.
                    Predict where the target car will be in the future.
                    Calculate how much the car will be in the 's' cooridnate.
                        a. prev_size = no of previous waypoints i.e time taken to reach the next waypoint
                        b. 0.02s - time increment for each step 
                        c. target_car_speed - Speed along the 's' coordinate
                        d. total distance = prev_size * 0.02 * target_car_speed
                        e. Add it to the current 's' coordinate of the target car */
              target_car_s += ((double)prev_size * 0.02 * target_car_speed);

              // Determine if target car is in the same lane and the distance between the target car
              // and the model car < 30 meters (unsafe for lane change), set the flag target_car_front
              if ((target_car_lane == lane) && (target_car_s > car_s) && ((target_car_s - car_s) < 30))
              {
                  target_car_front = true; // Target car front of model car
              }

              // Determine if target car is in the left lane and the distance between the target car
              // and the model car < 30 meters (unsafe for lane change), set the flag target_car_left
              else if ((target_car_lane == (lane - 1)) && (car_s - target_car_s < 30 && target_car_s - car_s < 30))
              {
                  target_car_left = true;
              }
              // Determine if target car is in the right lane and the distance between the target car
              // and the model car < 30 meters (unsafe for lane change), set the flag target_car_right
              else if ((target_car_lane == (lane + 1)) && (car_s - target_car_s < 30 && target_car_s - car_s < 30))
              {
                  target_car_right = true;
              }
          }

          // 3. Behavior Planning: Change the model vehicle's lane and velocity based on Target Vehicle's States
          
          // If target vehicle in front
          if (target_car_front)
          {
              // Reduce speed by 0.224. This satifies the design limits for acceleration and jerk 
              ref_vel -= 0.224;

              // If no target vehicle in the left lane, change to left lane
              if (lane > 0 && !target_car_left)
              {
                  lane--;
              }
              // If no target vehicle in the right lane, change to right lane
              else if (lane < 2 && !target_car_right) 
              {
                  lane++;
              }
          }
          // If no target vehicle in front
          else 
          {
              // If speed is less than speed limit, ramp up to speed limit
              if (ref_vel < 49.5) 
              {
                  ref_vel += .224;
              }

              // If we are not in middle lane, 
              if (lane != 1)
              {
                  // Check if middle lane (right lane) is empty if we are on the left lane or 
                  // Check if middel lane (left lane) is empty if we are on the right lane
                  // If empty change to middle lane
                  if ((lane == 0 && !target_car_right) || (lane == 2 && !target_car_left))
                  {
                      lane = 1;
                  }
              }
          }

              /* TRIAL CODE FROM CLASS
              * 
              if (d < (2 + 4 * lane + 2) && d >(2 + 4 * lane - 2))
              {
                  double vx = sensor_fusion[i][3];
                  double vy = sensor_fusion[i][4];
                  double check_speed = sqrt(vx * vx + vy * vy);
                  double check_car_s = sensor_fusion[i][5];

                  check_car_s += ((double)prev_size * 0.02 * check_speed); // If using previous points can project s value out
                  // Check s values greater than mine and s gap
                  if ((check_car_s > car_s) && ((check_car_s - car_s) < 30))
                  {
                      // Do some logic here, lower reference velocity so we dont crash into the car infront of us,
                      //could also flag to try to change lanes
                      //ref_vel = 29.5; //mph
                      too_close = true;
                      if (lane > 0)
                      {
                          lane = 0;
                      }

                  }
              }
          }

          if (too_close)
          {
              ref_vel -= .224;
          }
          else if (ref_vel < 49.5)
          {
              ref_vel += .224;
          }
          */
          //4. Trajectory Generation: Generate Trajectory for model car lane change procedure

          // Create a list of widely spaced (x,y) waypoints, evely spaced spaced at 30m/
          //Later we will interpolate these waypoints with a spline and fill it with points that control
          vector<double> ptsx;
          vector<double> ptsy;

          // Reference variables for x,y, yaw states. These will reference the starting point of the model car or
          // end point of the previous path
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);

          // Check if previous size is almost empty, use the model car as the starting reference
          if (prev_size < 2) 
          {
              // Use two points that make the path tangent to the car
              double prev_car_x = car_x - cos(car_yaw);
              double prev_car_y = car_y - sin(car_yaw);

              ptsx.push_back(prev_car_x);
              ptsx.push_back(car_x);

              ptsy.push_back(prev_car_y);
              ptsy.push_back(car_y);
          }

          //If previous path is not empty, use the previous path's end point as the starting reference
          else
          {
              //Redefine reference variables as previous path's end points
              ref_x = previous_path_x[prev_size - 1];
              ref_y = previous_path_y[prev_size - 1];

              double ref_x_prev = previous_path_x[prev_size - 2];
              double ref_y_prev = previous_path_y[prev_size - 2];
              //Calculate the reference yaw with the previous 2 points mentioned above
              ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

              // Use two points that make the path tangent to the previous path's end point
              ptsx.push_back(ref_x_prev);
              ptsx.push_back(ref_x);

              ptsy.push_back(ref_y_prev);
              ptsy.push_back(ref_y);

          }

          // In Frenet coordinates, add 3 evenly 30m spaced points ahead of the starting refernce
          vector<double> wp0 = getXY(car_s + 30, 2 + 4 * lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> wp1 = getXY(car_s + 60, 2 + 4 * lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> wp2 = getXY(car_s + 90, 2 + 4 * lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);

          // store newly created waypoints
          ptsx.push_back(wp0[0]);
          ptsx.push_back(wp1[0]);
          ptsx.push_back(wp2[0]);

          ptsy.push_back(wp0[1]);
          ptsy.push_back(wp1[1]);
          ptsy.push_back(wp2[1]);

          // Transform coordinates from inertial to model car's frame of reference
          for (int i = 0; i < ptsx.size(); i++)
          {
              // Shift Car reference angles to 0 deg
              double shift_x = ptsx[i] - ref_x;
              double shift_y = ptsy[i] - ref_y;

              // Transform to model car's reference frame
              ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw));
              ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
          }

          // Create a spline
          tk::spline s;

          // Set (x,y) waypoints to the spline
          s.set_points(ptsx, ptsy);

          // Define the actual (x,y) points we will use for the planner
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          // Add the previous path points to the planner
          //double dist_inc = 0.3;
          //for (int i = 0; i < 50; ++i)
          for (int i = 0; i < previous_path_x.size(); i++)
          {
              //double next_s = car_s + (i + 1) * dist_inc;
              //double next_d = 6;
              //vector<double> xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
              //next_x_vals.push_back(xy[0]);
              //next_y_vals.push_back(xy[1]);
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
          }

          // Calculate how to break the 30m spline so that we travel at our desired refernce velocity
          
          //Horizontal x-axis value 
          double target_x = 30.0;
          //Corresponding y-axis value
          double target_y = s(target_x);
          // Value of Hypotenuse
          double target_dist = sqrt((target_x) * (target_x)+(target_y) * (target_y));

          double x_add_on = 0;

          /* Fill up the rest of our path planner after filling it with previous points, here we are will always output 50 points.
          * There are two types of points. 1. Widely spaced spline points (x_point & y_point). These are not the previous path
          * points. 2. Previous path points which were filled previously into the planner in lines 362 - 371 */ 

          for (int i = 1; i <= 50 - previous_path_x.size(); i++)
          {
              // Calculate number of points to regulate speed. Here the distance between the two points is target_dist. It takes
              // 20ms to move from one point to another. ref_vel is in MPH so converting to m/s. Using the following formula
              // target_dist = N * 0.02 (ms) * ref_vel (MPH)/2.24 (conversion unit)
              double N = (target_dist / (0.02 * ref_vel / 2.24));
              // Using N to divide the x-axis to N points (target_x / N). Then add the number of points on x_axis to get x_point
              double x_point = x_add_on + (target_x) / N;
              // Calculate the corresponding y value for the x_point
              double y_point = s(x_point);

              // Add the x_point to calculate the x_points in the spline
              x_add_on = x_point;

              // Temp variables to convert back to global coordinates 
              double x_ref = x_point;
              double y_ref = y_point;

              // Transform back to global cooridnates
              x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
              y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

              // Add the calculated the x & y points to the initial x,y points of the model car from Localization
              x_point += ref_x;
              y_point += ref_y;

              //Push the final x,y points  to trajectory vectors in the path planner
              next_x_vals.push_back(x_point);
              next_y_vals.push_back(y_point);
          }
          // END CODE FOR PATH PLANNER

          json msgJson;

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}