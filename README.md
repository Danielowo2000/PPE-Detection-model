# PPE Detection using Streamlit

![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![GitHub stars](https://img.shields.io/github/stars/YourUsername/YourRepository.svg)](https://github.com/Danielowo2000/PPE-Detection-model/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/YourUsername/YourRepository.svg)](https://github.com/Danielowo2000/PPE-Detection-model/issues)

A real-time Personal Protective Equipment (PPE) detection application built with Streamlit and YOLOv8.

![Project Demo](./Picture1.png)

## Table of Contents

- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About

This project is designed to detect and highlight individuals wearing or not wearing specific PPE items, such as helmets, gloves, vests and boots. It utilizes the YOLOv8 model for object detection and is powered by Streamlit for the user interface. The application supports both webcam input and file upload for image and video analysis.

## Features

- Real-time PPE detection
- Web interface powered by Streamlit
- Supports webcam and file upload as input sources
- Adjustable confidence threshold
- Stop the webcam when not in use

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/Danielowo2000/PPE-Detection-model.git

2. Navigate to the project directory:

   ```sh
   cd YourRepository

3. Install the required Python packages:
   ```sh
   pip install -r requirements.txt

## Usage

1. To run the PPE detection application, use the following command:

   ```sh
   streamlit run app2.py

2. Open a web browser and access the Streamlit application at the provided URL (typically http://localhost:8501).
3. Select the input type (Webcam or File Upload) and set the confidence threshold.
4. Click the "Detect Objects" button to start object detection.
5. If using the webcam, you can click the "Stop Webcam" button to stop the webcam feed.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

