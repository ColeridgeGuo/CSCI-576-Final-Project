import math
import wave
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import scipy as sscc
import json
import os





#ZCR part
len_of_each_wav = []
path = r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\test_wav'
files= os.listdir(path)
waveFileAddress =[]
#lenOfFile = [4,5,4,6,5,6]
lenOfFile = [2,2,1,2,2,1]
zcrdata = []
ersbdata_average = []
rmsedata_average =[]
speccentdata_average = []
specbwdata_average = []
rolloffdata_average = []
zcrdata_average = []
volumedata_dr_average = []
volumedata_vstd_average = []
#ersbdata_std = []
#ersbdata_error = []



volumedata = []
for file in files:

    if not os.path.isdir(file):
        files_son = os.listdir(path + '\\' + str(file))
        temp = 'a'
        for f in files_son:
            if not os.path.isdir(file):
                if temp != file.split('_')[0]:
                    waveFileAddress.append(file.split('_')[0])
                    temp = file.split('_')[0]

print(waveFileAddress)





def get_wav_duration(audio_path):
    duration = librosa.get_duration(filename=audio_path)
    return duration


for len1 in range(0,len(lenOfFile)):
    temparray = []
    ersb_result_temparay = []
    ersb_avg_temparray = []


    rmse_avg_temparray = []
    speccent_avg_temparray = []
    specbw_avg_temparray = []
    rolloff_avg_temparray = []
    zcr_avg_temparray = []

    volumedr_avg_temparray = []
    volumevstd_avg_temparray = []





    volume_temparray = []
    video_num = 0


    for i in range(0,lenOfFile[len1]):
        video_num = i + 1
        x, sr = librosa.load(path +'\\'+ waveFileAddress[len1] +'\\'+waveFileAddress[len1] + "_" + str(i) + '.wav',sr=44100, duration=20)
        print(x.shape) #882000
        frame = librosa.util.frame(x, frame_length=1024, hop_length=512)
        #print(frame.shape)
        S = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, center=False))
        sc = librosa.feature.spectral_centroid(x, sr=sr, n_fft=1024, hop_length=512, center=False)
        ersb = np.zeros((4, S.shape[1]))  # (4,3045), this is energy ratio subbands
        print(S.shape)#1+1024/2 and 882000 / 512
        sub_band = [14, 39, 102, 512] # seperate whole wav file in 4 subband
        ersb_result = np.zeros(4) #this array store final average esrb

        rmse = librosa.feature.rms(x, frame_length=1024,hop_length=512,center=False)
        spec_cent = librosa.feature.spectral_centroid(x,n_fft=1024, hop_length=512,center=False)
        spec_bw = librosa.feature.spectral_bandwidth(x,n_fft=1024, hop_length=512,center=False)
        rolloff = librosa.feature.spectral_rolloff(x,n_fft=1024, hop_length=512,center=False)
        zcr = librosa.feature.zero_crossing_rate(x,frame_length=1024,hop_length=512,center=False)



        rmse_avg_temparray.append(sum(rmse[0]) / S.shape[1])
        speccent_avg_temparray.append(sum(spec_cent[0])/S.shape[1])
        specbw_avg_temparray.append(sum(spec_bw[0])/S.shape[1])
        rolloff_avg_temparray.append(sum(rolloff[0])/S.shape[1])
        zcr_avg_temparray.append(sum(zcr[0])/S.shape[1])
        dr_1sec = []
        vstd_1sec = []
        #zcr_1sec = []
        n0 = 0
        step = 45
        while n0 < 1721:
            end  = 0
            if n0< 1721:
                end = n0+step
            else:
                end = 1721
            dr_1sec.append((np.max(rmse[0][n0:end]) - np.min(rmse[0][n0:end]))/ np.max(rmse[0][n0:end]))
            vstd_1sec.append(np.std(rmse[0][n0:end])/np.max(rmse[0][n0:end]))
            #zcr_1sec.append(sum(zcr[0][n0:n0+step])/len(zcr[0][n0:n0+step]))

            n0 = end
        # (np.max(temppp) - np.min(temppp)) / np.max(temppp)
        volumedr_avg_temparray.append(sum(dr_1sec)/len(dr_1sec))
        volumevstd_avg_temparray.append(sum(vstd_1sec)/len(vstd_1sec))




        for j in range(S.shape[1]):
            ith_square_value = sum(np.power(S[:, j], 2))
            if ith_square_value > 0:
                ersb[0, j] = sum(np.power(S[:sub_band[0], j], 2)) / ith_square_value
                ersb[1, j] = sum(np.power(S[sub_band[0]:sub_band[1], j], 2)) / ith_square_value
                ersb[2, j] = sum(np.power(S[sub_band[1]:sub_band[2], j], 2)) / ith_square_value
                ersb[3, j] = sum(np.power(S[sub_band[2]:sub_band[3], j], 2)) / ith_square_value
            else:
                ersb[0, j] = 0
                ersb[1, j] = 0
                ersb[2, j] = 0
                ersb[3, j] = 0

        print(ersb.shape) # 4 X 1721
        #ersb_avg_temparray = ersb_avg_temparray + ersb
        #print(ersb_avg_temparray)
        #ersb_full_temparray.append(ersb)

        temparray2 = []
        for ersb_result_index in range(0,4):
            temparray2.append(float(format(sum(ersb[ersb_result_index]) / S.shape[1],'.6f')))
            #ersb_result_temparay.append(sum(ersb[ersb_result_index]) / ersb.shape[1])

        ersb_result_temparay.append(temparray2)
        #print(ersb_result_temparay)
        length = np.size(x)
        zero_crossings = librosa.zero_crossings(x, pad=False)
        zero_crossings = librosa.util.normalize(zero_crossings, axis=0)
        temparray.append(int(sum(zero_crossings)))

    #print(volumedr_avg_temparray)


    zcrdata.append(temparray)
    ersbdata_average.append(ersb_result_temparay)
    rmsedata_average.append(rmse_avg_temparray)
    speccentdata_average.append(speccent_avg_temparray)
    specbwdata_average.append(specbw_avg_temparray)
    rolloffdata_average.append(rolloff_avg_temparray)
    zcrdata_average.append(zcr_avg_temparray)
    volumedata_dr_average.append(volumedr_avg_temparray)
    volumedata_vstd_average.append(volumevstd_avg_temparray)


print(zcrdata)
print(ersbdata_average)
print(rmsedata_average)
print(speccentdata_average)
print(specbwdata_average)
print(rolloffdata_average)
print(zcrdata_average)
print(volumedata_dr_average)
print(volumedata_vstd_average)



audio_name = []
big_dict_zcrcount = {}
big_dict_rmse = {}
big_dict_speccent = {}
big_dict_specbw = {}
big_dict_rolloff = {}
big_dict_zcr = {}
big_dict_volume_dr = {}
big_dict_volume_vstd = {}

big_dict_ersb_sub1 = {}
big_dict_ersb_sub2 = {}
big_dict_ersb_sub3 = {}
big_dict_ersb_sub4 = {}
for index in range(0,6):
    temp_name = []
    for index2 in range(0,len(zcrdata[index])):
        temp_str = waveFileAddress[index] + '_' + str(index2)
        temp_name.append(temp_str)
    audio_name.append(temp_name)

print(audio_name)

ii = 0
for iii in range(0,len(audio_name)):
    d1 = zip(audio_name[iii],zcrdata[iii])
    d3 = zip(audio_name[iii], rmsedata_average[iii])
    d4 = zip(audio_name[iii], speccentdata_average[iii])
    d5 = zip(audio_name[iii], specbwdata_average[iii])
    d6 = zip(audio_name[iii], rolloffdata_average[iii])
    d7 = zip(audio_name[iii], zcrdata_average[iii])
    d8 = zip(audio_name[iii], volumedata_dr_average[iii])
    d9 = zip(audio_name[iii], volumedata_vstd_average[iii])

    t1 = []
    t2 = []
    t3 = []
    t4 = []
    for jjj in range(0, len(ersbdata_average[iii])):
        t1.append(ersbdata_average[iii][jjj][0])
        t2.append(ersbdata_average[iii][jjj][1])
        t3.append(ersbdata_average[iii][jjj][2])
        t4.append(ersbdata_average[iii][jjj][3])

    d2_1 = zip(audio_name[iii], t1)
    d2_2 = zip(audio_name[iii], t2)
    d2_3 = zip(audio_name[iii], t3)
    d2_4 = zip(audio_name[iii], t4)

    big_dict_zcrcount[waveFileAddress[iii]] = dict(d1)
    big_dict_rmse[waveFileAddress[iii]] = dict(d3)
    big_dict_speccent[waveFileAddress[iii]] = dict(d4)
    big_dict_specbw[waveFileAddress[iii]] = dict(d5)
    big_dict_rolloff[waveFileAddress[iii]] = dict(d6)
    big_dict_zcr[waveFileAddress[iii]] = dict(d7)
    big_dict_volume_dr[waveFileAddress[iii]] = dict(d8)
    big_dict_volume_vstd[waveFileAddress[iii]] = dict(d9)

    big_dict_ersb_sub1[waveFileAddress[iii]] = dict(d2_1)
    big_dict_ersb_sub2[waveFileAddress[iii]] = dict(d2_2)
    big_dict_ersb_sub3[waveFileAddress[iii]] = dict(d2_3)
    big_dict_ersb_sub4[waveFileAddress[iii]] = dict(d2_4)

# big_dict_zcrcount[waveFileAddress[iii]] = dict(d1)
# big_dict_rmse[waveFileAddress[iii]] = dict(d3)
# big_dict_speccent[waveFileAddress[iii]] = dict(d4)
# big_dict_specbw[waveFileAddress[iii]] = dict(d5)
# big_dict_rolloff[waveFileAddress[iii]] = dict(d6)
# big_dict_zcr[waveFileAddress[iii]] = dict(d7)
# big_dict_volume_dr[waveFileAddress[iii]] = dict(d8)
# big_dict_volume_vstd[waveFileAddress[iii]] = dict(d9)
#
# big_dict_ersb_sub1[waveFileAddress[iii]] = dict(d2_1)
# big_dict_ersb_sub2[waveFileAddress[iii]] = dict(d2_2)
# big_dict_ersb_sub3[waveFileAddress[iii]] = dict(d2_3)
# big_dict_ersb_sub4[waveFileAddress[iii]] = dict(d2_4)

dict1 = {}
dict3 = {}
dict4 = {}
dict5 = {}
dict6 = {}
dict7 = {}
dict8 = {}
dict9 = {}
dict2_1 = {}
dict2_2 = {}
dict2_3 = {}
dict2_4 = {}

dict1['feature name'] = 'zcr total count'
dict1['value'] = dict(big_dict_zcrcount)

dict3['feature name'] = 'Root mean square energy(volume)'
dict3['value'] = dict(big_dict_rmse)

dict4['feature name'] = 'Spectral centroid'
dict4['value'] = dict(big_dict_speccent)

dict5['feature name'] = 'Spectral Bandwidth'
dict5['value'] = dict(big_dict_specbw)

dict6['feature name'] = 'Spectral Rolloff'
dict6['value'] = dict(big_dict_rolloff)

dict7['feature name'] = 'Zero Crossing Rate'
dict7['value'] = dict(big_dict_zcr)

dict8['feature name'] = 'Volume Dynamic Range'
dict8['value'] = dict(big_dict_volume_dr)

dict9['feature name'] = 'Volume Standard deviation'
dict9['value'] = dict(big_dict_volume_vstd)

dict2_1['feature name'] = 'Energy Ration in subband (0-630) HZ'
dict2_1['value'] = dict(big_dict_ersb_sub1)

dict2_2['feature name'] = 'Energy Ration in subband (631-1720) HZ'
dict2_2['value'] = dict(big_dict_ersb_sub2)

dict2_3['feature name'] = 'Energy Ration in subband (1721-4400) HZ'
dict2_3['value'] = dict(big_dict_ersb_sub3)

dict2_4['feature name'] = 'Energy Ration in subband (4400-22000) HZ'
dict2_4['value'] = dict(big_dict_ersb_sub4)


json_str1 = json.dumps(dict1, indent=4)
json_str3 = json.dumps(dict3, indent=4)
json_str4 = json.dumps(dict4, indent=4)
json_str5 = json.dumps(dict5, indent=4)
json_str6 = json.dumps(dict6, indent=4)
json_str7 = json.dumps(dict7, indent=4)
json_str8 = json.dumps(dict8, indent=4)
json_str9 = json.dumps(dict9, indent=4)
json_str2_1 = json.dumps(dict2_1, indent=4)
json_str2_2 = json.dumps(dict2_2, indent=4)
json_str2_3 = json.dumps(dict2_3, indent=4)
json_str2_4 = json.dumps(dict2_4, indent=4)

with open('zcrcount.json', 'w') as json_file:
    json_file.write(json_str1)
with open('rmse.json', 'w') as json_file:
    json_file.write(json_str3)
with open('centroid.json', 'w') as json_file:
    json_file.write(json_str4)
with open('bandwidth.json', 'w') as json_file:
    json_file.write(json_str5)
with open('rollof.json', 'w') as json_file:
    json_file.write(json_str6)
with open('zcr.json', 'w') as json_file:
    json_file.write(json_str7)
with open('volumedr.json', 'w') as json_file:
    json_file.write(json_str8)
with open('volumestd.json', 'w') as json_file:
    json_file.write(json_str9)
with open('ersb1.json', 'w') as json_file:
    json_file.write(json_str2_1)
with open('ersb2.json', 'w') as json_file:
    json_file.write(json_str2_2)
with open('ersb3.json', 'w') as json_file:
    json_file.write(json_str2_3)
with open('ersb4.json', 'w') as json_file:
    json_file.write(json_str2_4)



# print(dict(big_dict_zcrcount))
# print(dict(big_dict_rmse))
# print(dict(big_dict_speccent))
# print(dict(big_dict_specbw))
# print(dict(big_dict_rolloff))
# print(dict(big_dict_zcr))
# print(dict(big_dict_volume_dr))
# print(dict(big_dict_volume_vstd))
# print(dict(big_dict_ersb_sub1))
# print(dict(big_dict_ersb_sub2))
# print(dict(big_dict_ersb_sub3))
# print(dict(big_dict_ersb_sub4))

# dit = {}
# for i in d:
#     dit[i[0]] = i[1]
# print(dit)

# std = [[0.023865844902749993, 0.0030090697411875003, 0.011407347413187498, 0.0014719444461875], [0.004638504556159998, 0.0008841926005600005, 0.0016461678506399999, 0.000249834736], [0.0022917051785000015, 0.0021683846892500006, 0.0001101194895, 1.2275360187500003e-05], [0.007124545428555558, 0.007843394779000002, 0.0016768148453333335, 0.0007079562311388889], [0.011607973084559998, 0.008340955494239998, 0.0008027565927999998, 0.0002425144052], [0.014446515652187499, 0.0035884230591874972, 0.0042080396690000015, 3.374269349999999e-05]]
# error = [[0.011932922451374997, 0.0015045348705937502, 0.005703673706593749, 0.00073597222309375], [0.002074402300303249, 0.00039542295201089593, 0.000736188643281152, 0.00011172949056734277], [0.0011458525892500007, 0.0010841923446250003, 5.505974475e-05, 6.137680093750001e-06], [0.002908583491539937, 0.0032020525099599397, 0.0006845567940317604, 0.0002890219210856909], [0.005191243379612814, 0.003730188696484198, 0.00035900366217738356, 0.0001084557391100257], [0.007223257826093749, 0.0017942115295937486, 0.0021040198345000007, 1.6871346749999995e-05]]
#
#
# ercb_a =[[0.6147064999999999, 0.17954275, 0.12888925, 0.07686125], [0.6401998, 0.2437878, 0.09700460000000001, 0.019008], [0.788441, 0.18758249999999999, 0.021767, 0.0022092500000000003], [0.7131416666666666, 0.179409, 0.061941, 0.045508166666666676], [0.7696951999999999, 0.1456116, 0.06032900000000001, 0.024364], [0.35541125, 0.44926275000000004, 0.165269, 0.030057]]
# x_axis = ['0~630','631~1720','1721~4400','4401~22000']
# name = ['ads','cartoon','concerts','interview','movies','sport']
# color = ['b','g','r','y','k','b']
# shape = ['.-','.-','.-','.-','.-','*--']
# plt.figure('Energy ratio in subband: average', figsize = (14,5))
# for line in range(0,6):
#     plt.plot(x_axis,ercb_a[line],color[line] + shape[line],label=name[line])
#
# plt.xlabel('subband(HZ)')
# plt.xticks(x_axis)
# plt.ylabel('Energy ration in subband')
# plt.legend()
#
# plt.figure('Energy ratio in subband: std', figsize = (14,5))
# for line in range(0,6):
#     plt.plot(x_axis,std[line],color[line] + shape[line],label=name[line])
#
# plt.xlabel('subband(HZ)')
# plt.xticks(x_axis)
# plt.ylabel('Energy ration in subband')
# plt.legend()
#
# plt.figure('Energy ratio in subband: error', figsize = (14,5))
# for line in range(0,6):
#     plt.plot(x_axis,error[line],color[line] + shape[line],label=name[line])
#
# plt.xlabel('subband(HZ)')
# plt.xticks(x_axis)
# plt.ylabel('Energy ration in subband')
# plt.legend()
#
# plt.grid()
# plt.show()





# for i in range(0,len(zcrdata)):
#     result = int(sum(zcrdata[i]) / len(zcrdata[i]))
#     print(result)










#ads
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\ads\ads_0.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\ads\ads_1.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\ads\ads_2.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\ads\ads_3.wav','rb')


#Cartoon
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\cartoon\cartoon_0.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\cartoon\cartoon_1.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\cartoon\cartoon_2.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\cartoon\cartoon_3.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\cartoon\cartoon_4.wav','rb')

#interview
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\interview\interview_0.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\interview\interview_1.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\interview\interview_2.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\interview\interview_3.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\interview\interview_4.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\interview\interview_5.wav','rb')

#movies
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\movies\movies_0.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\movies\movies_1.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\movies\movies_2.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\movies\movies_3.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\movies\movies_4.wav','rb')

#sport
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\sport\sport_0.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\sport\sport_1.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\sport\sport_2.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\sport\sport_3.wav','rb')


#concerts
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\concerts\concerts_0.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\concerts\concerts_1.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\concerts\concerts_2.wav','rb')
#fw = wave.open(r'C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_wav\concerts\concerts_3.wav','rb')





