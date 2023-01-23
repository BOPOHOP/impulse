GS-PLOT V(0.1) 2023/01/21
--------------
Python based MCA for gamma spectrometry.
----------------------------------------
 This program reads a stream of data from the PC sound card and picks out pulses, the pulses are subsequently filtered and written to a JSON file. A live histogram can be viewed in the browser. 
 
 Installation
 ------------
 This is a beta development version and still unfinished. 
 To run the program, download the gs_plot folder, then navigate to the code folder from your terminal and type in at the command prompt; 
 
 ~ %  Python3 run.py
 
You may need to install the latest version of python3 and various other dependencies. 

When it's all working you can access the program in your browser at;

http://localhost:8050
 

Things that are working
-----------------------

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


Things to do
------------
* Build polynomial pulse height calibration and save calibration data
* Build method for subtracting a background spectrum
* Build function and chart to display distortion curve (useful)
* Build isotope peak tables 
* Improve look and layout
* Build interval histogram with Dead time calculation 

If anyone has requests for additional features please contact me via the "Contact us" link at gammaspectacular.com


Steven Sesselmann

Gammaspectacular.com
