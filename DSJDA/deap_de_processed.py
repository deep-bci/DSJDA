import os
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def read_file(file):
	data = sio.loadmat(file)
	data = data['data']
	# print(data.shape)
	return data

def compute_DE(signal):
	variance = np.var(signal,ddof=1)
	return math.log(2*math.pi*math.e*variance)/2

def decompose(file):
	# trial*channel*sample
	start_index = 384 #3s pre-trial signals
	data = read_file(file)
	shape = data.shape
	frequency = 128

	decomposed_de = np.empty([0,5,60])

	base_DE = np.empty([0,160])
	# (delta 1-4 Hz, theta 4-7 Hz, alpha 8-14 Hz, beta 14-31Hz, and gamma 31-45 Hz
	for trial in range(40):
		temp_base_DE = np.empty([0])
		temp_base_delta_DE = np.empty([0])
		temp_base_theta_DE = np.empty([0])
		temp_base_alpha_DE = np.empty([0])
		temp_base_beta_DE = np.empty([0])
		temp_base_gamma_DE = np.empty([0])

		temp_de = np.empty([0,60])

		for channel in range(32):
			trial_signal = data[trial,channel,384:]
			base_signal = data[trial,channel,:384]
			#****************compute base DE****************
			base_delta = butter_bandpass_filter(base_signal, 1, 4, frequency, order=3)
			base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
			base_alpha = butter_bandpass_filter(base_signal, 8,14, frequency, order=3)
			base_beta = butter_bandpass_filter(base_signal,14,31, frequency, order=3)
			base_gamma = butter_bandpass_filter(base_signal,31,45, frequency, order=3)

			base_delta_DE = (compute_DE(base_delta[:128]) + compute_DE(base_delta[128:256]) + compute_DE(base_delta[256:])) / 3
			base_theta_DE = (compute_DE(base_theta[:128])+compute_DE(base_theta[128:256])+compute_DE(base_theta[256:]))/3
			base_alpha_DE =(compute_DE(base_alpha[:128])+compute_DE(base_alpha[128:256])+compute_DE(base_alpha[256:]))/3
			base_beta_DE =(compute_DE(base_beta[:128])+compute_DE(base_beta[128:256])+compute_DE(base_beta[256:]))/3
			base_gamma_DE =(compute_DE(base_gamma[:128])+compute_DE(base_gamma[128:256])+compute_DE(base_gamma[256:]))/3

			temp_base_delta_DE = np.append(temp_base_delta_DE, base_delta_DE)
			temp_base_theta_DE = np.append(temp_base_theta_DE,base_theta_DE)
			temp_base_gamma_DE = np.append(temp_base_gamma_DE,base_gamma_DE)
			temp_base_beta_DE = np.append(temp_base_beta_DE,base_beta_DE)
			temp_base_alpha_DE = np.append(temp_base_alpha_DE,base_alpha_DE)

			delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
			theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
			alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
			beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
			gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

			DE_delta = np.zeros(shape=[0], dtype=float)
			DE_theta = np.zeros(shape=[0],dtype = float)
			DE_alpha = np.zeros(shape=[0],dtype = float)
			DE_beta =  np.zeros(shape=[0],dtype = float)
			DE_gamma = np.zeros(shape=[0],dtype = float)

			for index in range(60):
				DE_delta = np.append(DE_delta, compute_DE(delta[index * frequency:(index + 1) * frequency]))
				DE_theta =np.append(DE_theta,compute_DE(theta[index*frequency:(index+1)*frequency]))
				DE_alpha =np.append(DE_alpha,compute_DE(alpha[index*frequency:(index+1)*frequency]))
				DE_beta =np.append(DE_beta,compute_DE(beta[index*frequency:(index+1)*frequency]))
				DE_gamma =np.append(DE_gamma,compute_DE(gamma[index*frequency:(index+1)*frequency]))
			temp_de = np.vstack([temp_de,DE_delta])
			temp_de = np.vstack([temp_de,DE_theta])
			temp_de = np.vstack([temp_de,DE_alpha])
			temp_de = np.vstack([temp_de,DE_beta])
			temp_de = np.vstack([temp_de,DE_gamma])
		temp_trial_de = temp_de.reshape(-1,5,60)
		decomposed_de = np.vstack([decomposed_de,temp_trial_de])

		temp_base_DE = np.append(temp_base_delta_DE,temp_base_theta_DE)
		temp_base_DE = np.append(temp_base_DE,temp_base_alpha_DE)
		temp_base_DE = np.append(temp_base_DE,temp_base_beta_DE)
		temp_base_DE = np.append(temp_base_DE,temp_base_gamma_DE)
		base_DE = np.vstack([base_DE,temp_base_DE])
	decomposed_de_1 = decomposed_de.reshape(-1,32,5,60).transpose([0,3,2,1]).reshape(-1,5,32).reshape(-1,128)
	decomposed_de_2 = decomposed_de.reshape(-1, 32, 5, 60).transpose([0, 3, 1, 2]).reshape(-1, 32, 5).transpose(1, 0, 2)
	print("base_DE shape:",base_DE.shape)
	print("trial_DE_1 shape:",decomposed_de_1.shape)
	print("trial_DE_2 shape:", decomposed_de_2.shape)
	return base_DE,decomposed_de_1,decomposed_de_2

def get_labels(file):
	#0 valence, 1 arousal, 2 dominance, 3 liking
	valence_labels = sio.loadmat(file)["labels"][:,0]>5	# valence labels
	arousal_labels = sio.loadmat(file)["labels"][:,1]>5	# arousal labels
	final_valence_labels = np.empty([0])
	final_arousal_labels = np.empty([0])
	for i in range(len(valence_labels)):
		for j in range(0,60):
			final_valence_labels = np.append(final_valence_labels,valence_labels[i])
			final_arousal_labels = np.append(final_arousal_labels,arousal_labels[i])
	print("labels:",final_arousal_labels.shape)
	return final_arousal_labels,final_valence_labels

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data. nonzero ()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	return data_normalized


if __name__ == '__main__':
	dataset_dir = "yourpath"

	result_dir = "yourpath"
	if os.path.isdir(result_dir)==False:
		os.makedirs(result_dir)

	for file in os.listdir(dataset_dir):
		print("processing: ",file,"......")
		file_path = os.path.join(dataset_dir,file)
		base_DE,trial_DE_1,trial_DE_2 = decompose(file_path)
		arousal_labels,valence_labels = get_labels(file_path)
		sio.savemat(result_dir+"DE_"+file,{"base_data":base_DE,"data_1":trial_DE_1,"data_2":trial_DE_2,"valence_labels":valence_labels,"arousal_labels":arousal_labels})
