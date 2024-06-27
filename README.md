# GestuNova

GestuNova is an application that recognizes hand gestures using MediaPipe and performs corresponding Linux system commands. This project utilizes MediaPipe for hand landmark detection and a machine learning model to classify the detected landmarks into predefined hand signs or gestures. The recognized gestures are then mapped to specific commands, which can emulate keyboard key presses, run applications, or execute terminal commands. Additionally, users can run Python scripts, unlocking numerous automation possibilities.

## Features

- Real-time hand gesture recognition using MediaPipe
- Logging and preprocessing of hand landmark coordinates
- Classification of hand gestures using a machine learning model
- Mapping of gestures to Linux system commands
- Execution of terminal commands, including keyboard emulation and application launches
- Support for running custom Python scripts
- Ability to change and train new gesture classes

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/YOUR_USERNAME/GestuNova.git
    cd GestuNova
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```sh
    python gui.py
    ```

2. Use the GUI to bind specific commands to gestures. You can specify commands such as opening applications, controlling the volume, navigating through tabs, and running custom Python scripts.

## Examples of Commands

Here are some example commands you can assign to gestures:

- Open Google Chrome:
    ```sh
    google-chrome
    ```
- Increase volume:
    ```sh
    amixer -D pulse sset Master 10%+
    ```
- Decrease volume:
    ```sh
    amixer -D pulse sset Master 10%-
    ```
- Next tab:
    ```sh
    xdotool key ctrl+Tab
    ```
- Previous tab:
    ```sh
    xdotool key ctrl+shift+Tab
    ```
- Next song:
    ```sh
    playerctl next
    ```
- Previous song:
    ```sh
    playerctl previous
    ```
- Run a Python script:
    ```sh
    python /path/to/script.py
    ```

## Custom Python Scripts

GestuNova can execute custom Python scripts, allowing for endless automation possibilities. Simply specify the path to your Python script in the command field, and the script will be executed when the corresponding gesture is recognized.

## Changing and Training Gesture Classes

GestuNova allows you to change the predefined gesture classes and even train new ones using the methods provided in the original repository. If the default model does not meet your requirements, you can log your own data and train the model accordingly.

### Steps to Change and Train Gesture Classes

1. **Log Your Own Data:** Follow the instructions in the original repository to capture and log new gesture data. This involves recording the hand landmarks for the desired gestures.

2. **Train the Model:** Use the logged data to train a new model. The original repository provides a step-by-step guide on how to preprocess the data, configure the training, and train the model.

3. **Update GestuNova:** Replace the existing model with your newly trained model in the GestuNova application. Update the gesture-to-command mappings in the GUI to reflect the new gesture classes.

For detailed instructions, refer to the documentation and guides provided in the original repository from which this project was forked.

## Acknowledgments

This project builds on the work of the following individuals. Their contributions to the backend of the application are greatly appreciated.

- **Author:** Kazuhito Takahashi [Twitter](https://twitter.com/KzhtTkhs)
- **Translation and other improvements:** Nikita Kiselov [GitHub](https://github.com/kinivi)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
