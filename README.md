# 🚁 6-DOF Drone Simulator — INDI Control

A real-time **6-DOF quadrotor simulation** developed in Python, combining nonlinear drone dynamics, **Incremental Nonlinear Dynamic Inversion (INDI)** control, motor allocation, and interactive 3D visualization.

The project was developed to study and implement a practical flight-control architecture while keeping the simulation and controller fully transparent and customizable.

![Drone Simulator](6-DOF%20INDI%20Drone%20Simulator%202026-02-25%2012_48_08%20AM.png)

---

## 🎯 Project Overview

The simulator models a quadrotor as a rigid-body system with:

* 3D position and velocity
* Quaternion-based attitude representation
* Angular velocity
* Four rotor dynamics
* Thrust and reaction torque
* Aerodynamic drag
* Wind disturbance
* Ground effect
* Gyroscopic rotor effects

The control architecture combines an **outer attitude loop**, an **inner INDI angular-rate controller**, and an **altitude PD controller**.

---

## 🧠 Control Architecture

```text
                   User Commands
                         │
                         ▼
              ┌─────────────────────┐
              │   Attitude Command   │
              │ Roll / Pitch / Yaw  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Attitude Loop     │
              │   Rate Reference    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    INDI Controller  │
              │   Angular Rate Loop │
              └──────────┬──────────┘
                         │
                    Δ Torque
                         │
                         ▼
              ┌─────────────────────┐
              │  Control Allocation │
              │   Motor Mixing      │
              └──────────┬──────────┘
                         │
                    Motor Speeds
                         │
                         ▼
              ┌─────────────────────┐
              │   Drone Dynamics    │
              │  6-DOF Simulation   │
              └──────────┬──────────┘
                         │
                         ▼
              Position / Attitude
                         │
                         └──────► Feedback
```

---

## ⚙️ Main Features

### Flight Dynamics

The dynamics model includes:

* Translational dynamics
* Rotational dynamics
* Quaternion attitude propagation
* Rotor motor dynamics
* Thrust generation
* Reaction torque
* Gyroscopic effects
* Aerodynamic drag
* Wind
* Ground effect

The numerical integration is implemented using a **fourth-order Runge-Kutta (RK4)** method.

### INDI Controller

The attitude controller uses a cascaded structure:

**Attitude error → Desired angular rate → Angular acceleration → Incremental torque → Motor thrust**

The INDI controller estimates measured angular acceleration from consecutive angular-rate measurements and computes an incremental torque correction.

The controller also includes:

* Rate feedback
* Attitude feedback
* Torque saturation
* Thrust constraints
* Control allocation
* Motor-speed conversion
* Thrust-priority handling

### Altitude Control

A separate PD controller regulates the altitude:

```text
Altitude Error
      ↓
   PD Control
      ↓
 Total Thrust
      ↓
  INDI + Mixer
      ↓
 Motor Commands
```

The altitude target can be adjusted during simulation.

---

## 🖥️ Interactive 3D Simulation

The simulator includes a real-time OpenGL visualization with:

* 3D quadrotor model
* Animated propellers
* World coordinate axes
* Ground grid
* Drone shadow
* Target altitude indicator
* Altitude gauge
* Artificial horizon
* Attitude indicators
* Camera orbit control

The simulation runs at a control timestep of **5 ms (200 Hz)**.

---

## 🎮 Controls

| Key     | Function                            |
| ------- | ----------------------------------- |
| `W / S` | Pitch                               |
| `A / D` | Roll                                |
| `Q / E` | Yaw                                 |
| `↑ / ↓` | Increase / decrease altitude target |
| `← / →` | Orbit camera                        |

---

## 📁 Project Structure

```text
drone INDI/
│
├── control.py
│   └── INDI controller and control allocation
│
├── drone.py
│   └── Simulation loop and OpenGL visualization
│
├── dronedynamics.py
│   └── Quadrotor physical model and dynamics integration
│
├── drone_technical_latex.pdf
│   └── Technical documentation
│
└── 6-DOF INDI Drone Simulator.png
    └── Simulation screenshot
```

---

## 🛠️ Technologies

* **Python**
* **NumPy**
* **OpenGL / PyOpenGL**
* **GLFW**
* Numerical simulation
* Quaternion-based rigid-body dynamics
* INDI control
* RK4 integration

---

## 🚀 Running the Simulation

Install the required Python packages:

```bash
pip install numpy PyOpenGL glfw
```

Then run:

```bash
python drone.py
```

Make sure `control.py` and `dronedynamics.py` are in the same directory as `drone.py`.

---

## 📊 Engineering Concepts Demonstrated

This project demonstrates practical implementation of:

* Nonlinear system modeling
* 6-DOF rigid-body dynamics
* Quaternion attitude representation
* Flight control
* Incremental Nonlinear Dynamic Inversion
* Cascaded control architecture
* Control allocation
* Motor dynamics
* Numerical integration
* Real-time simulation
* 3D visualization

---

## 📚 Technical Documentation

A detailed technical document is included in:

`drone_technical_latex.pdf`

It provides additional mathematical and engineering details behind the simulation and control design.

---

## 🔧 Future Improvements

Possible extensions include:

* Trajectory tracking
* Position control
* Wind/disturbance testing
* Sensor noise and state estimation
* IMU simulation
* PID vs INDI performance comparison
* Control-performance metrics
* Automated trajectory generation
* Real-time telemetry plots
* Hardware-in-the-loop simulation
* PX4 / ArduPilot integration

---

## 👨‍💻 Author

**Mohamed Salah Dahassa**

Automation & Control Engineer

Interested in:

`Industrial Automation` · `Robotics` · `Control Systems` · `Flight Control` · `Simulation` · `Intelligent Systems`

---

## ⭐ Project Highlights

**6-DOF Dynamics • INDI Control • Motor Allocation • Real-Time Simulation • OpenGL Visualization**
