import numpy as np
from scipy.signal import butter, filtfilt


# Define high-pass filter parameters
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# Split the data into smaller segments
def Split_Sequence(data,n_stpes):
    x=[]
    for i in range(0,len(data),n_stpes):
        end=i+n_stpes
        if end>len(data)-1:
            break
        x_seq=data[i:end]      
        x.append(x_seq)  
    x=np.array(x)
    return x


def load_spikes(dpi):

    # Prepare data
    import read_data
    data_ct,data_dn,data_zk=read_data2.read_each_class(dpi)

    n_step=836
    data_ct=Split_Sequence(data_ct,n_step)
    data_dn=Split_Sequence(data_dn,n_step)
    data_zk=Split_Sequence(data_zk,n_step)

    label_ct=np.zeros((data_ct.shape[0]))
    label_dn=np.ones((data_dn.shape[0]))
    label_zk=np.ones((data_zk.shape[0]))*2

    raw_signal=np.concatenate((data_ct, data_dn, data_zk))
    labels=np.concatenate((label_ct, label_dn, label_zk))
    print('Raw signal shape: ',raw_signal.shape)


    # Apply the high-pass filter to the raw signal
    cutoff_frequency = 700 
    sampling_rate = 30000   
    filtered_signal = highpass_filter(raw_signal, cutoff_frequency, sampling_rate)
    print('filtered signal shape: ',filtered_signal.shape)

    # Extract spikes
    spike_data=[]
    for item in filtered_signal:
        new_array1 = np.where((item > 0) & (item <= 10), 0, item)
        new_array2 = np.where((new_array1 < 0) & (new_array1 >= -10), 0, new_array1)
        spike_data.append(new_array2)
    spike_data=np.array(spike_data)
    print('Spike data shape: ',spike_data.shape)

    return spike_data, labels

