# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---
In this project, a PID controller is developed and tuned. It is implemented on a simulation car to follow a particular path by adjusting the steering angle.
## Reflection

#### PID Controller
PID stands for Proportional Integral Derivative. It is one of the most popular controllers used in control loop feedback mechanism. The controller continuously calculates the error function of the parameter that needs to be controlled (in this project it is the distance from the centre line). It then applies a control input (in this project it is the steering angle) that is based on the proportional(P), integral(I) and derivative(D) of the error term for correction

#### PID Parameters
1. P (Proportional) Gain Component: This gain regulates how large the control input (steering angle) for a given error would be. Hence this gain component applies a steering angle proportional to the Cross Track Error (CTE). If the car is to the left of the centre line, it steers right and vice versa. During this maneuver it overshoots and leads to an oscillatory motion. Hence if this gain is too high, the system becomes unstable
2. D (Derivative) Gain Component: This gain regulates the rate of change of the CTE. This indicates that if the CTE is low i.e. the desired point is reached, it smoothens the control input to reduce the overshoot and oscillations caused by the P controller. Hence if this value is too low, the osicallations might be high. This components improves system stability and settling time.
3. I(Integral) Gain Component: This gain regulates the sum of all the CTE's at a point for a given time frame. It eliminates the steady state error caused by PD controller. It also accounts for any systematic biases.

The PID controllers were manually tuned. The driving behavior of the car with respect to the parameter changes were analyzed from the simulator. The final tuned parameters were selected that allowed the car to drive smoothly around the lap without popping over ledges or lane lines. The final values are:
P - 0.2
I - 0.0004
D - 2.0

#### Manual Tuning Procedure
The manual tuning procedure is as follow:
1. Set all the PID gains to 0.
2. Increase the P gain (Kp) until the cte is small and the oscillations are minimal. Also ensure it moves to the desired location fairly fast
3. Increase the D gain (Kd) to remove the overshooting and the oscillations. If Kd is too high, it will lead to overshooting again
4. Increase the I gain (Ki) to remove any steady state error. Suggest startiung with a very small number (0.0001)
5. Tune around the gains to get the best performance
## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

Fellow students have put together a guide to Windows set-up for the project [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Kidnapped_Vehicle_Windows_Setup.pdf) if the environment you have set up for the Sensor Fusion projects does not work for this project. There's also an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3).

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/e8235395-22dd-4b87-88e0-d108c5e5bbf4/concepts/6a4d8d42-6a04-4aa6-b284-1697c0fd6562)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

