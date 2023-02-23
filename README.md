IMPULSE Version 1.1 February 2023
--------------
Python Scripted MCA for gamma spectrometry.
----------------------------------------
 This program reads a stream of data from the PC sound card and picks out pulses, the pulses are subsequently filtered and written to a JSON file. A live histogram can be viewed in the browser. 
 
 Installation
 ------------
Step 1)
Download the package from the Github repository here 

Step 2) 
Unzip the package to the preferred location on your computer 

Step 3)
If not already installed on your computer, download and install the latest version of Python 3 from the official site, consider upgrading if you are on a very old version ... www.python.org

Step 4)
Open your terminal to the command line and navigate to the folder impulse-main

Step 5) 
It is necessary to install some python libraries required for Impulse, so copy and paste the following into your terminal;
pip install -r requirements.txt

Step 6) 
If you are using macOS or iOS, you also need to install PyObjC. To do this, copy and paste the following into your terminal:
pip install -r requirements_macos.txt

Step 7) 
Navigate up to the code directory \impulse\code\ by typing cd code

Step 8) 
Now run the program by typing python run.py, mac users may have to type "python3 run.py"
python run.py Fingers crossed your default browser should open up and show tab 1

Step 9) 
Always exit the program from tab 4 by clicking the exit button (important)

When it's all working you can access the program in your browser at;

http://localhost:8050
 

Change log
------------------------

0) creating a settings database in sqlite3
1) Obtaining an indexed table of audio devices 
2) Selecting Audio input device
3) Reading the audio stream and finding pulses
4) Function to find the average sample vanues in a list of n pulses
5) Function to normalise a pulse
6) Function to save the normalised pulse as a csv file (needs changing to JSON)
7) Browser layout with tabs
8) Tab for editing settings and capturing pulse shape
9) Graph to display captured pulse
10) Function to calculate pulse distortion
11) Function to read data stream, find pulses and append counts to histogram
12) Function to save histogram in JSON format
13) Tab for displaying pulse height histogram and filter settings
14) Assigned program name "impulse"
15) Tidy up and move all styling to assets/styles.css
16) Added function to show spectrum in log scale
17) Added polynomial pulse height calibration and save calibration data
18) Completed method for subtracting a background spectrum
19) Added 482 isotope peak libraries in json format
20) Program now auto detects negative pulse direction and inverts samples
21) Modified devise list to only show input devices
22) Wrapped shape function in a try: except: in case no audio device is available.
23) Fixed an issue where the x axis changed scale when comparison was switched on.
24) Added peakfinder function with resolution notation (Bug prevents notation showing in log scale)
25) Added new function and chart to display distortion curve (useful) 

Things to do
------------
* Build interval histogram with Dead time calculation 
* Add browse button for isotope comparisons
* Build option for switching theme between fun/boring :)

If anyone has requests for additional features please contact me via the "Contact us" link at gammaspectacular.com


Steven Sesselmann

Gammaspectacular.com

