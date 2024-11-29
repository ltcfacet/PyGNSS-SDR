# PyGNSS-SDR
https://catedras.facet.unt.edu.ar/labtel/es/home/

This repository contains a Python-based software tool for GPS signal acquisition based on SDR devices, like the HackRF One and RTL-SDR. 

## Installation

1. Install virtualenv:

    pip install virtualenv

    virtualenv --version

2. Create a virtual environment:

    virtualenv --python=/usr/bin/python3 my-env         # Create a virtual environment
    
3. Activate the vitual environment:

    source my-env/bin/activate                          # Activate the virtual environment 

4. Install Dependencies

    pip install -r requirements.txt                     # Install the dependencies

5. To deactivate the virtual environment:

    deactivate

6. To remove the virtual environment:

    rm -rf ./my-env 

## Running the PyGNSS-SDR

Navigate to the project folder and run ./menu.py
(If you encounter permission issues, you need to grant the operating system permission to execute it using the command: chmod +x menu.py)

## Test File

You can download two files with real recordings to test the tool from the following link:
https://mega.nz/folder/xmQSQLgb#m9kiAv4NpG_h22O3mr6Nbw

1. capture1_gnss_sdr.dat
    * GNSS-SDR Format
    * Sampling Frequency: 2e6
    * Satellites: 2,12,24,29

2. capture2_rtl_sdr.dat
    * RTL-SDR Format
    * Sampling Frequency: 2e6
    * Satellites: 2,12,25,29


