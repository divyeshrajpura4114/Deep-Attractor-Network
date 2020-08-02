from .libraries import *
from . import params, utils

class features():
    def __init__(self):
        super(features, self).__init__()
        self.hp = params.Hparam().load_hparam()

    def extract_features(self):
        file_list = glob.glob(os.path.join(self.hp.data.dataset_path, self.hp.data.data_folder, self.hp.features.data_type, 'mixture', '*.wav'))

        if not os.path.exists(os.path.join(self.hp.data.dataset_path, self.hp.data.feature_folder, self.hp.features.seq_type, self.hp.features.data_type)):
            os.makedirs(os.path.join(self.hp.data.dataset_path, self.hp.data.feature_folder, self.hp.features.seq_type, self.hp.features.data_type))

        for file_index in range(len(file_list)):
            file_path = file_list[file_index]
            mix_features, mix_abs, mix_ph, weight_thresh, ideal_mask, vocal_abs, vocal_ph, inst_abs, inst_ph = self.spectrogram(file_path, self.hp.features.data_type)
            
            if self.hp.features.seq_type == 'complete':
                self.seq_complete(file_path, self.hp.features.data_type, mix_features, mix_abs, mix_ph, weight_thresh, ideal_mask, vocal_abs, vocal_ph, inst_abs, inst_ph)
            elif self.hp.features.seq_type == 'partial':
               self.seq_partial(file_path, self.hp.features.data_type, mix_features, mix_abs, mix_ph, weight_thresh, ideal_mask, vocal_abs, vocal_ph, inst_abs, inst_ph)
            else:
                print('Please provide valid sequcence type')


    def spectrogram(self, file_path_full, data_type):
        file_path, file_name = os.path.split(file_path_full)
        print(file_name)
        
        mix_features, mix_abs, mix_ph, mix_power = self.get_mel_stft(os.path.join(file_path, file_name))
        vocal_features, vocal_abs, vocal_ph, vocal_power = self.get_mel_stft(os.path.join(file_path.replace('mixture','vocal'),file_name))
        inst_features, inst_abs, inst_ph, inst_power = self.get_mel_stft(os.path.join(file_path.replace('mixture','instrument_mixture'),file_name))

        vocal_wfm = np.divide(vocal_power, vocal_power + inst_power)
        inst_wfm = np.divide(inst_power, vocal_power + inst_power)
        
        vocal_wfm = vocal_wfm.T
        inst_wfm = inst_wfm.T
        vocal_wfm.resize((vocal_wfm.shape[0],vocal_wfm.shape[1],1))
        inst_wfm.resize((inst_wfm.shape[0],inst_wfm.shape[1],1))
        wfm = np.concatenate((vocal_wfm,inst_wfm),axis = 2)
        
        # threashold = np.min(mix_power) + 0.005 * (np.max(mix_power) - np.min(mix_power))
        # mix_weights = (mix_power > threashold).astype(int)
        weight_thresh = np.ones(shape = mix_power.shape)
        
        return mix_features.T, mix_abs.T, mix_ph.T, weight_thresh.T, wfm, vocal_abs.T, vocal_ph.T, inst_abs.T, inst_ph.T

    def get_stft(self, file_path):
        s_raw, _ = librosa.core.load(file_path, sr = self.hp.data.sr)
        s_spec = librosa.core.stft(s_raw, n_fft=self.hp.features.nfft, 
                                    hop_length=self.hp.features.hop_size, 
                                    win_length = self.hp.features.window_size, 
                                    window = np.sqrt(np.hanning(self.hp.features.window_size)))
        
        s_spec = s_spec + float(self.hp.features.eps)
        s_abs = np.abs(s_spec)
        s_ph = np.angle(s_spec)
        s_power = np.power(s_abs,2)
        s_log = np.log(s_abs)
        return s_log, s_abs, s_ph, s_power

    def get_mel_stft(self,file_path):
        s_raw, _ = librosa.core.load(file_path, sr = self.hp.data.sr)
        s_stft = librosa.core.stft(s_raw, n_fft = self.hp.features.nfft, 
                                    hop_length = self.hp.features.hop_size, 
                                    win_length = self.hp.features.window_size, 
                                    window = np.sqrt(np.hanning(self.hp.features.window_size)))
        s_stft = s_stft + float(self.hp.features.eps)
        s_abs = np.abs(s_stft)
        s_ph = np.angle(s_stft)
        mel = librosa.filters.mel(sr = self.hp.data.sr, n_fft = self.hp.features.nfft, n_mels = self.hp.features.nmels)
        s_mel_stft = mel.dot(s_abs)
        s_mel_power = np.power(s_mel_stft,2)
        return s_mel_stft, s_abs, s_ph, s_mel_power

    def seq_complete(self, file_path, data_type, mix_features, mix_abs, mix_ph, weight_thresh, ideal_mask, vocal_abs, vocal_ph, inst_abs, inst_ph):
        file_path, file_name = os.path.split(file_path)
        file_name = os.path.splitext(file_name)[0]

        np.savez(os.path.join(self.hp.data.dataset_path, self.hp.data.feature_folder, 'complete',data_type, file_name), 
                    mix_features = mix_features, mix_abs = mix_abs, mix_ph = mix_ph, weight_thresh = weight_thresh, ideal_mask = ideal_mask,
                    vocal_abs = vocal_abs, vocal_ph = vocal_ph, inst_abs = inst_abs, inst_ph = inst_ph)

    def seq_partial(self,file_path, data_type, mix_features, mix_abs, mix_ph, weight_thresh, ideal_mask, vocal_abs, vocal_ph, inst_abs, inst_ph):
        file_path, file_name = os.path.split(file_path)
        file_name = os.path.splitext(file_name)[0]

        seq_length = self.hp.features.seq_length

        for index in range(int(mix_features.shape[0]/seq_length)):
            mix_abs_updated = mix_abs[index*seq_length : (index+1)*seq_length,:]
            mix_ph_updated = mix_ph[index*seq_length : (index+1)*seq_length,:]
            mix_features_updated = mix_features[index*seq_length : (index+1)*seq_length,:]
            
            vocal_abs_updated = vocal_abs[index*seq_length : (index+1)*seq_length,:]
            vocal_ph_updated = vocal_ph[index*seq_length : (index+1)*seq_length,:]

            inst_abs_updated = inst_abs[index*seq_length : (index+1)*seq_length,:]
            inst_ph_updated = inst_ph[index*seq_length : (index+1)*seq_length,:]
            
            ideal_mask_updated = ideal_mask[index*seq_length : (index+1)*seq_length,:,:]
            ideal_mask_updated = ideal_mask_updated.reshape(ideal_mask_updated.shape[0]*ideal_mask_updated.shape[1],ideal_mask_updated.shape[2])

            weight_thresh_updated = weight_thresh[index*seq_length : (index+1)*seq_length,:]

            np.savez(os.path.join(self.hp.data.dataset_path, self.hp.data.feature_folder,'partial',data_type, file_name + '_' +str(index)), 
                    mix_features = mix_features_updated, mix_abs = mix_abs_updated, mix_ph = mix_ph_updated, 
                    weight_thresh = weight_thresh_updated, ideal_mask = ideal_mask_updated,
                    vocal_abs = vocal_abs_updated, vocal_ph = vocal_ph_updated, inst_abs = inst_abs_updated, inst_ph = inst_ph_updated)