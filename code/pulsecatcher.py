# This page is the main pulse catcher file, it 
# collects, normalises and filters the pulses 
# ultimately saving the histogram file to JSON.
import pyaudio
import wave
import math
import time
import functions as fn
import sqlite3 as sql
import datetime
from collections import defaultdict
import csv
import numpy as np

data 			= None
left_channel 	= None
path 			= None
device_list 	= fn.get_device_list()
plot 			= {}

global_cps      = 0
global_counts	= 0
grand_cps	= 0
read_size	= 0

# Function reads audio stream and finds pulses then outputs time, pulse height and distortion
def pulsecatcher(mode):

	# Start timer
	t0				= datetime.datetime.now()
	tb				= time.time()	#time beginning
	tla = 0

	# Get the following from settings
	settings 		= fn.load_settings()
	filename        = settings[1]
	device          = settings[2]             
	sample_rate     = settings[3]
	chunk_size      = settings[4]                        
	threshold       = settings[5]
	tolerance       = settings[6]
	bins            = settings[7]
	bin_size        = settings[8]
	max_counts      = settings[9]
	sample_length	= settings[11]
	coeff_1			= settings[18]
	coeff_2			= settings[19]
	coeff_3			= settings[20]
	flip 			= settings[22]
	max_seconds     = settings[26]
	t_interval      = settings[27]
	peakshift       = settings[28]

	peak 		    = int((sample_length-1)/2) + peakshift
	condition       = True

	# Create an array of empty bins
	start 			= 0
	stop 			= bins * bin_size
	histogram 		= [0] * bins
	histogram_3d 	= [0] * bins
	audio_format 	= pyaudio.paInt16
	device_channels = fn.get_max_input_channels(device)

	# Loads pulse shape from csv
	shapestring = fn.load_shape()

	# Converts string to float
	shape 		= [int(x) for x in shapestring]
	samples 	= []
	pulses 		= []
	left_data 	= []

	p = pyaudio.PyAudio()

	global global_cps
	global global_counts  

	print(tolerance)

	global_cps 		= 0
	global_counts 		= 0
	interval_counts 	= 0
	elapsed 		= 0


	elapsed 		= 0
	grand_cps 		= 0
	read_size 		= 0

	if max_counts < 100:
		max_counts = 111111000

	# Open the selected audio input device
	stream = p.open(
		format   			= audio_format,
		channels    		= device_channels,
		rate  				= sample_rate,
		input  				= True,
		output  			= False,
		input_device_index  = device,
		frames_per_buffer   = chunk_size * 2)

	tla = time.time()
	read_size = 0
	rest = [ ]
	delta_h_sum = 0
	h_mult_sum = 0.
	dist_sum = 0.
	rejected_counts = 0
	while condition and (global_counts < max_counts and elapsed <= max_seconds):
		# Read one chunk of audio data from stream into memory. 
		data = stream.read(chunk_size, exception_on_overflow=False)
		# Convert hex values into a list of decimal values
		values = list(wave.struct.unpack("%dh" % (chunk_size * device_channels), data))
		# Extract every other element (left channel)
		left_channel = values[::2]
		read_size += len(left_channel)
		# Flip inverts all samples if detector pulses are positive
		if flip != 1:
			left_channel = [flip * x for x in left_channel]

		left_channel = rest + left_channel
		skip_to = 0

		# Read through the list of left channel values and find pulse peaks
		for i, sample in enumerate(left_channel[:-sample_length]):
			if i < skip_to:
				continue
			# iterate through one sample lenghth at the time in quick succession, ta-ta-ta-ta-ta...
			samples = left_channel[i:i+sample_length]
			# Function calculates pulse height of all samples 
			# height = fn.pulse_height(samples)
			# Filter out noise
			if ((s_max := samples[peak]) == max(samples) 
#q#					and (height := fn.pulse_height_q2(peak, samples)) > threshold 
					and (height := samples[peak] - (s_min := min(samples))) > threshold 
					and samples[peak] < 32768):
				# Function normalises sample to zero and converts to integer
				normalised = fn.normalise_pulse_min_max(s_min, s_max, samples)
				# Compares pulse to sample and calculates distortion number
				distortion = fn.distortion(normalised, shape)

				if distortion < tolerance:
					# Filters out distorted pulses
					# advance next analyze pos to current + sample_length
					# skip_to = i + sample_length - 1
					skip_to = i + int(sample_length * 4 / 5)



#07#					# h_mult =  distortion * 0.0000025	#d05 0.0000025/0.02 ok 7.6% 121 391 865
					# h_mult =  distortion * 0.0000050	#d09 0.0000020/0.015 ok 7.6% 121 391 865
					# h_mult =  distortion * 0.0000050 #d11 0.0000050/0.015 ok 7.6% 142/459/1015 / 2048/10/740v 100/3500
#d#					h_mult =  distortion * 0.0000030 #d11 0.0000050/0.015 ok 7.6% 142/459/1015 / 2048/10/740v 100/3500
					# h_mult =  distortion * 0.0000030 #d?? 0.0000030 ok ...% 141/457/1013 d:1100 / 2048/10/740v 100/3500
#n#					if h_mult > 0.030:
#n#						h_mult = 0.030

					# h_add = .000075  * distortion * samples[peak] # bad..
					# height = samples[peak] + h_add - min(samples)

#DD#					h_mult =  distortion * 2.2e-10 #d13 2.2e-10/**2 ok 7.7% 141/457/1005 2048/10/740v d:50/110000000
#DD#					h_mult =  distortion * 5.4e-6 #d13 2.2e-10/**2 ok 7.7% 141/457/1005 2048/10/740v d:50/110000000
					h_old = height
					height = fn.pulse_height_q2(peak, s_min, samples)
					h_mult = 0
					if height > 0:
						h_mult = (height - h_old) / height
#DD#					height *= (1 + h_mult)



					# Sorts pulse into correct bin
					bin_index = int(height/bin_size)

					# Adds 1 to the correct bin
					if bin_index < bins:
						histogram[bin_index] 	+= 1
						histogram_3d[bin_index] += 1 
						global_counts  		+= 1	
						global_cps 		+= 1
						interval_counts 	+= 1

#Q1#						h_mult = (height - samples[peak] + s_min) / height
						delta_h = int(height/bin_size) - int(h_old/bin_size)
						delta_h_sum += delta_h
						h_mult_sum += h_mult
						dist_sum += distortion


				else: # distortion < tolerance:
					rejected_counts += 1

		rest = left_channel[i+1:]

		t1      = datetime.datetime.now() # Time capture
		te      = time.time()
		elapsed = te - tb
		if elapsed > 0:
			grand_cps = global_counts / elapsed
		else:
			grand_cps = 0

		# Saves histogram to json file at interval
		if te - tla >= t_interval:
			settings 		= fn.load_settings()
			filename        = settings[1]
			max_counts      = settings[9]
			max_seconds		= settings[26]
			coeff_1			= settings[18]
			coeff_2			= settings[19]
			coeff_3			= settings[20]

			global_cps = int(global_cps/(te-tla))
			
			if mode == 2:
				fn.write_histogram_json(t0, t1, bins, global_counts, int(elapsed), filename, histogram, coeff_1, coeff_2, coeff_3)
				fn.write_histogram_csv(t0, t1, bins, global_counts, int(elapsed), filename, histogram, coeff_1, coeff_2, coeff_3, read_size, elapsed)

			if mode == 3:
				fn.write_3D_intervals_json(t0, t1, bins, global_counts, int(elapsed), filename, histogram_3d, coeff_1, coeff_2, coeff_3)
				histogram_3d = [0] * bins

			tla = time.time()
			reject_percent = 0
			if global_counts != 0 and elapsed != 0:
				reject_percent = rejected_counts/global_counts * 100;
			if interval_counts != 0 and elapsed != 0:
				d1 = dist_sum/interval_counts
				d2 = h_mult_sum/interval_counts*100
				delta_h_avg = delta_h_sum / interval_counts
			else:
				d1 = 0
				d2 = 0
				delta_h_avg = 0

			print("elapsed=%4d cps=%7.2f reject_cps=%6.2f %6.2f%% rate=%.2f dist_avg = %10.2f h_mult_avg = %6.2f%% dh=%4.1f" % (
					elapsed, 
					global_counts/elapsed, rejected_counts/elapsed, reject_percent,
					read_size/elapsed/1000,
					d1,
					d2,
					delta_h_avg
					))

			h_mult_sum = 0.
			dist_sum = 0.
			delta_h_sum = 0.

			fn.write_cps_json(filename, global_cps)
			global_cps = 0
			interval_counts = 0
	
	p.terminate() # closes stream when done
	print("pulsecatcher stop")
	return						
											
