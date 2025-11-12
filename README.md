# DroneProject
The Autonomous Drone Parking Monitoring System is a comprehensive solution designed to automate the process of monitoring and reporting real-time parking occupancy in large car parks or facilities.

Purpose
The primary purpose is to increase parking efficiency and reduce driver frustration by replacing traditional, fixed ground sensors with a flexible, mobile platform. It provides facility managers with accurate, actionable data for operational improvement.

Mechanism
The system operates using an Autonomous Drone Platform equipped with a camera and an onboard computer (Raspberry Pi).

The drone flies a pre-programmed mission path over the car park.

An onboard Computer Vision (AI) algorithm analyzes the captured images at the "edge" to classify parking bays as occupied or free.

The drone transmits the compiled occupancy results wirelessly via a Wi-Fi Communication Link to a central Ground Station.

The Ground Station updates a User Interface (GUI) to display a clear, real-time visual map of available spaces.
