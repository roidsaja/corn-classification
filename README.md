# WEB-BASED UTILITY FOR THE CLASSIFICATION OF CORN PLANT DISEASES USING PRE-TRAINED DEEP LEARNING ARCHITECTURE

A thesis prototype project for IULI in determining a corn plant disease by the use of state-of-the-art architecture called EfficientNet. This repository is a sign of progression in developing the prototype before it will be released out into the open public for real world uses.

# Setup
1. Activate the virtualenv **venv** by inserting the following:
    ```
    $ source venv/bin/activate
    ```
2. Execute Streamlit locally by inserting the following:
    ```
    $ streamlit run main.py
    ```
3. Head over to the given url link or let the execution of streamlit open them for you on a new browser tab:
  ![image](https://user-images.githubusercontent.com/22772929/164340370-7f9a7201-1fbe-45f4-821c-822782e5d425.png)

4. Start classifying by selecting saved images:
  ![image](https://user-images.githubusercontent.com/22772929/164340526-486ef3f2-bee8-4510-b435-43104de491d5.png)

# Mac M1 Processor Setup
1. Have homebrew installed and miniforge installed as part of the `brew` installation process.
2. Initialize the virtualenv:
    ```
    $ conda create --prefix ./mac-env python=3.8
    ```
3. Activating and Deactivating the virtualenv:
    ```
    $ conda activate ./mac-env
    ```
    ```
    $ conda deactivate
    ```
4. Install TensorFlow dependencies from Apple Conda channel:
    ```
    $ conda install -c apple tensorflow-deps
    ```
5. Install base TensorFlow:
    ```
    $ python -m pip install tensorflow-macos
    ```
6. Install TensorFlow Metal from Apple to leverage Apple's M1, M1 Pro and M1 Max GPU Acceleration:
    ```
    $ python -m pip install tensorflow-metal
    ```
7. (Optional) Install TensorFlow datasets for benchmarking:
    ```
    $ python -m pip install tensorflow-datasets
    ```
8. Install the used data science packages:
    ```
    $ conda install jupyter pandas numpy matplotlib scikit-learn pillow
    ```
    ```
    $ pip install split-folders streamlit
    ```