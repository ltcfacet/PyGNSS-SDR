#!/usr/bin/env python3      
#Permite correr directamente con ./script.py si se le da permido chmod +x script.py
# -*- coding: utf-8 -*-

"""
Created on Fri Sep 20 10:48:31 2019

GPS_L1_CA_PCPS_Acquisition
Parallel Code Phase Search Acquisition

@author: Lucas E. CASTILLO DELACROIX
"""

import os
import subprocess as sp
from multiprocessing import Pool, cpu_count
from time import time
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft,fftfreq,fftshift,ifftshift,get_workers,set_workers   #permite multithread
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.tri.triangulation import Triangulation
from mpl_toolkits.mplot3d.axis3d import ZAxis


file = None
format_file = None
search_option = None 
satellite = None 
sample_rate = None 
analysis_time = None 
coherent_integration_time = None
number_dwells = None 
dwell_time = None
max_dwells = None 
doppler_step = None 
doppler_max = None 
doppler_range = None 
bin_chip = None 
treshold = None
dump_file = False
graphic = False
sats = {}                                                           #diccionario que almacena las fft de los codigos prn
freq_list = None

#Funcion de Lectura de Archivo

def read(file_name, sample_rate, coherent_time, ndwells, total_time):

    samples_coherent = int(coherent_time*sample_rate)

    # GNSS-SDR FORMAT FILE (gr-complex)
    if format_file == 1 :
        packdef = '<2f'
        bytes_per_complex_sample = 8
        signed_correct_factor = 0
        normalize_factor = 1
    
    # LIB RTL-SDR FORMAT FILE (gr-complex)
    if format_file == 2 :
        packdef = '<2B'
        bytes_per_complex_sample = 2
        signed_correct_factor = -127.5-127.5j
        normalize_factor = 127.5

    with open(file_name, 'rb') as rawfile:

        for t in range(total_time):

            for i in range(ndwells):

                array = []
                #print('Posicion del indice del archivo(en bytes) a leer: ',rawfile.tell())       #devuelve la posicion del puntero de posicion del archivo

                for j in range(samples_coherent):
                    array.append((complex(*struct.unpack(packdef, rawfile.read(bytes_per_complex_sample)))+signed_correct_factor)/normalize_factor)

                #print('Integracion {} lee {} muestras'.format(i+1,len(array)))
                #print('Posicion del indice del archivo(en bytes) despues de leer: ',rawfile.tell())
                yield np.array(array)

            rawfile.seek((t+1)*bytes_per_complex_sample*int(sample_rate),0)                                          #permite saltear al sgte segundo de datos sin leer muestras. reposiciona el puntero de posicion del archivo 1s mas adelante.


#Funciones de generacion y extension de codigos

SV = {
   1: [2,6],
   2: [3,7],
   3: [4,8],
   4: [5,9],
   5: [1,9],
   6: [2,10],
   7: [1,8],
   8: [2,9],
   9: [3,10],
  10: [2,3],
  11: [3,4],
  12: [5,6],
  13: [6,7],
  14: [7,8],
  15: [8,9],
  16: [9,10],
  17: [1,4],
  18: [2,5],
  19: [3,6],
  20: [4,7],
  21: [5,8],
  22: [6,9],
  23: [1,3],
  24: [4,6],
  25: [5,7],
  26: [6,8],
  27: [7,9],
  28: [8,10],
  29: [1,6],
  30: [2,7],
  31: [3,8],
  32: [4,9],}

def shift(register, feedback, output):
    """GPS Shift Register
    
    :param list feedback: which positions to use as feedback (1 indexed)
    :param list output: which positions are output (1 indexed)
    :returns output of shift register:
    
    """
    
    # calculate output
    out = [register[i-1] for i in output]
    if len(out) > 1:
        out = sum(out) % 2
    else:
        out = out[0]
        
    # modulo 2 add feedback
    fb = sum([register[i-1] for i in feedback]) % 2
    
    # shift to the right
    for i in reversed(range(len(register[1:]))):
        register[i+1] = register[i]
        
    # put feedback in position 1
    register[0] = fb
    
    return out

def PRN(sv):
    """Build the CA code (PRN) for a given satellite ID
    
    :param int sv: satellite code (1-32)
    :returns list: ca code for chosen satellite
    
    """
    
    # init registers
    G1 = [1 for i in range(10)]
    G2 = [1 for i in range(10)]

    ca = [] # stuff output in here
    
    # create sequence
    for i in range(1023):
        g1 = shift(G1, [3,10], [10])
        g2 = shift(G2, [2,3,6,8,9,10], SV[sv]) # <- sat chosen here from table
        
        # modulo 2 add and append to the code
        ca.append((g1 + g2) % 2)

    # return C/A code!
    return [-1 if x==0 else 1 for x in ca]

def sampler_extender(sv, sample_rate, coherent_time):
    """descrete sample a PRN for a satellite
    """

    samples_coherent = int(coherent_time*sample_rate)
    chiping_rate = 1.023e6  # From GPS Spec
    prn = PRN(sv)
    samples = []

    for i in range(samples_coherent):
        t = i / float(sample_rate)
        p = prn[int(round(t*chiping_rate))%1023]
        samples.append(complex(p, 0))

    return np.array(samples)

    # test1 = sampler_extender(satellite, sample_rate, coherent_integration_time)
    # print(len(test1))
    # print(test1)
    # plt.figure()
    # #plt.subplot(2,1,1)
    # plt.step(range(len(test1)),test1)

def prealocation(sample_rate, coherent_time):

    global sats

    if (search_option == 1 ):
        
        sats = {}

        for sv in range(1,33,1):
            prn_time = sampler_extender(sv, sample_rate, coherent_time)
            sats[sv] = fft(prn_time)
            #sats[sv] = fftshift(fft(prn_time))     #prueba solucion signo
            #print(len(prn_time))
            #print(prn_time)

    elif (search_option == 2):

        sats = {}
        prn_time = sampler_extender(satellite, sample_rate, coherent_time)
        sats[satellite] = fft(prn_time)
        #sats[satellite] = fftshift(fft(prn_time))      #prueba solucion signo


#Funciones de PDS y Analisis

def test(sv, fft_signal_incoming, doppler_freq_test, doppler_step):
    """test a sample of data using a cross-correlation technique for a signal

    :param int sample_rate: how fast the data was sampled
    :param data: numpy complex array of data
    :param int SV: test satellite number
    :param float doppler: test doppler shift
    """
    correlation = []

    fft_test = np.roll(fft_signal_incoming, int(doppler_freq_test/doppler_step))

    correlation = np.multiply(fft_test, np.conj(sats[sv])) 

    return np.square(np.absolute(ifft(correlation)))
    #return np.square(np.absolute(ifftshift(correlation)))       #prueba cambio de signo

def acquisition(sat, data, umbral):

    frequencies_correlation = []
    max_snr = 0
    min_snr = 1000

    for doppler_freq in doppler_range:
        
        cross_correlation = test(sat, data, doppler_freq, doppler_step)

        #plt.figure(doppler_freq)
        #plt.plot(cross_correlation)

        # Compute stats about this test
        tot_pwr = 0
        snr = [0 for i in range(1023 * bin_chip)]

        for i, pwr in enumerate(cross_correlation):
            # bin by code phase
            phase = int(i * (1.023e6*bin_chip) / sample_rate % (1023*bin_chip))
            tot_pwr += pwr
            snr[phase] += pwr

        # Normaliza a la potencial total
        snr_norm = np.divide(snr, tot_pwr)  #ojo devuele valores floats

        # Calula max y min normalizado
        max_norm_snr = max(snr_norm)
        min_norm_snr = min(snr_norm)

        #Chequea los umbrales y calculando el maximo de todas las frecuencias
        if max_snr < max_norm_snr:
            max_snr = max_norm_snr
        if min_snr > min_norm_snr:
            min_snr = min_norm_snr
            
        frequencies_correlation.append(snr_norm)

        # print("\nFrecuencia: ",doppler_freq)
        # print("SNR Max: ", max_snr)
        # print("SNR Min: ", min_snr)

    correlation_matrix = np.array(frequencies_correlation)      #Convierto lista de correlaciones en matriz

    acq, freq, delay = detection (sat, correlation_matrix, umbral)        #Devuelve la posicion de los maximos de la matriz de correlacion (los devuelve siempre, pero solo tiene sentido leerlos cuando acq_st=1) 

    #print("\nBuscando Satelite {}\n".format(sat))
    #print("\nFrecuencias Procesadas: ",len(frequencies_correlation))
    #print('\nMatriz de Correlacion u: ',correlation_matrix.shape,'\n\n', correlation_matrix)

    return correlation_matrix, acq, freq, delay    

def detection(sat, correlation_matrix, umbral):

    acquisition_st = 0

    index_max = np.unravel_index(np.argmax(correlation_matrix, axis=None), correlation_matrix.shape)
    max = round(correlation_matrix[index_max],5)

    #doppler_shift = -doppler_max+index_max[0]*doppler_step             #Funcion original
    doppler_shift = -(-doppler_max+index_max[0]*doppler_step)           #Trucar signo frecuencia mostrada
    #doppler_shift = freq_list[index_max[0]]                            # intento de acomodar signo

    delay_chip = index_max[1]/bin_chip

    if(max > umbral):

        acquisition_st = 1

        print('\nSATELLITE {} VISIBLE\n'.format(sat))
        print('Frecuencia de Desplazamiento Doppler: ', doppler_shift)
        print('Delay Code [chips]: ', delay_chip)
        #print('Maximo: ', max)
        
    return acquisition_st, doppler_shift, delay_chip

def check_max_dwells_acquistions(acq_matrix_state):

    acq_counter_list = []
    acq_sat_state = []

    for i in range(len(acq_matrix_state)):

        acq_counter_list.append(sum(acq_matrix_state[i][:]))


    for i, acquisitions in enumerate(acq_counter_list):

        if(acquisitions >= max_dwells):

            acq_sat_state.insert(i,1)
            
            if(search_option == 1):
                indice = i+1
            elif(search_option == 2):
                indice = satellite
            
            print('\nSATELLITE {} LISTO PARA DECODIFICAR\n'.format(indice))

        else:
            acq_sat_state.insert(i,0)

    return acq_counter_list, acq_sat_state


#Funciones de Almacenamiento y Graficos

seconds = 0                                                           #only for update plot title

def update_plot(frame_number, zarray, X, Y, surface, cbar, title, ax):
    
    global seconds
    #fig.delaxes(cbar.ax)
    #fig.delaxes(fig.axes[1])
    #cbar.ax.collections.clear()
    #cbar.ax.clear()
    
    if((frame_number%number_dwells)==0): seconds+=1
    title.set_text("Satellite {} Search Space - Second {} - Integration {}".format(satellite,seconds-1,(frame_number%number_dwells)+1))
    ax.collections.clear()

    surface = ax.plot_surface(X, Y, zarray[frame_number], cmap='jet', rstride=1, cstride=1, linewidth=0, alpha=1)     #autumn_r/jet/hot/afmhot
    #cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label='Correlation Power')
    maximo=round(zarray[frame_number].max(),4)
    cbar.set_ticklabels([0, maximo/2, maximo])
    
    #print('Frame: {}'.format(frame_number))

    return surface,title,cbar

def dump(sat, second, matrix_non_coherent, doppler_shift, delay_chip, acquisition_matrix_st, acquisition_count, acquisition_sat_st):

    dump_filename = 'acquisition_sat_{}_second_{}.txt'.format(sat, second+1)
    #print('Generando archivo: ', dump_filename)

    if(search_option == 1):
        indice = sat-1
    elif(search_option == 2):
        indice = 0

    with open('./acquisition/' + dump_filename, 'w') as archivo:

        
        archivo.write('\nTiempo total de analisis: {} s'.format(analysis_time))
        archivo.write('\nTiempo de integracion coherente: {} s'.format(coherent_integration_time))
        archivo.write('\nNumero de integraciones por cada ventana no coherente: {}'.format(number_dwells))
        archivo.write('\nTiempo de integracion no coherente: {} s'.format(dwell_time))
        archivo.write('\nRango de busqueda Frecuencial: ± {} Hz'.format(doppler_max))
        archivo.write('\nResolucion del bin Frecuencial: {} Hz'.format(doppler_step))
        archivo.write('\nResolucion del bin chip: {}'.format(1/bin_chip))
        archivo.write('\nSatelite buscado: {}'.format(sat))
        archivo.write('\nUmbral de deteccion: {}'.format(treshold))
        archivo.write("\nNumero de adquisiciones positivas por cada ventana no coherente para lograr adquisicion: {}".format(max_dwells))
        archivo.write('\nCantidad de Adquisiciones positivas: {}'.format(acquisition_count[indice]))
        archivo.write('\nEstado de Adquisicion: {}'.format(acquisition_sat_st[indice]))
        archivo.write('\nMatriz de Correlaciones(t={}): {}'.format(second+1, np.array(matrix_non_coherent[indice]).shape))    
            
        for i, matrix_coherent in enumerate(matrix_non_coherent[indice]):

            salto = '\n\n  #########################################################################################################################################\n  ###################################################          INTEGRACION {}          #####################################################\n  #########################################################################################################################################\n\n\n'.format(i+1)
            archivo.write(salto)
            archivo.write('Estado de adquisicion para esta integracion: {}'.format(acquisition_matrix_st[indice][i]) + '\n\n')
            
            if(acquisition_matrix_st[indice][i] == 1):
                archivo.write('\nFrecuencia de Desplazamiento Doppler: {}'.format(doppler_shift[indice][i]))
                archivo.write('\nDelay Code [chips]: {}'.format(delay_chip[indice][i]) + '\n\n')

            np.savetxt(archivo, matrix_coherent, fmt='%.5f', comments='', delimiter =', ', newline = '\n\n\n')


#Funcion Principal

def main():

    tiempo_inicial=time()
    global seconds
    seconds = 0
    
    #Creacion de la carpeta para los archivos txt de adquisicion
    if (dump_file):
        
        if(os.path.isdir('./acquisition')):
            sp.run(['rm','-r', 'acquisition'])

        sp.run(['mkdir', 'acquisition'])

    data_generator = read(file, sample_rate, coherent_integration_time, number_dwells, analysis_time)    #Objeto generador
    prealocation(sample_rate, coherent_integration_time)
    correlation_matrix_temporal = []

    #Analisis por cada segundo
    for t in range(analysis_time):

        print('\n\nTiempo Analizado: {} s'.format(t+1))

        correlation_matrix_non_coherent = []                                    
        acquisitions_matrix_state = []
        doppler_freqs = []
        delay_codes = []

        #Analisis en cada ventan de integracion no coherente
        for n in range(number_dwells):

            print('\n\nIntegracion {}'.format(n+1))
            #print('\nNuevos Datos')

            data_actual = next(data_generator)
            fftdata = fft(data_actual,norm="ortho")
            #fftdata = fftshift(fft(data_actual,norm="ortho"))       #prueba cambio de signo
            
            
            #print('Nro de hilos: ', get_workers())
            
            # with set_workers(4):
            #     print('Nro de hilos: ', get_workers())
            #     fftdata = fft(data_actual,norm="ortho")

            with Pool(cpu_count() - 1) as p:

                resultado = p.starmap(acquisition, [(sv,fftdata,treshold) for sv in sats])


            for correlation_matrix, adq, doppler_freq, delay_code in resultado:
                
                correlation_matrix_non_coherent.append(correlation_matrix)
                acquisitions_matrix_state.append(adq)
                doppler_freqs.append(doppler_freq)
                delay_codes.append(delay_code)

                correlation_matrix_temporal.append(correlation_matrix)


        #print('\nLista de Tuplas (matrix, adq_st, freq, code) con todos los datos para cada satelite: ', np.array(resultado).shape)

        #(sat*n_dwells, freq, code)
        correlation_matrix_non_coherent = np.array(correlation_matrix_non_coherent)
        #print('\nMatriz de matrices de correlacion no coherentes sin formatear: ', correlation_matrix_non_coherent.shape)

        #(sat,non_coherent_integration(n_dwells),freq,code)
        correlation_matrix_non_coherent = correlation_matrix_non_coherent.reshape(len(sats), correlation_matrix_non_coherent.shape[0]//len(sats), correlation_matrix_non_coherent.shape[1], correlation_matrix_non_coherent.shape[2], order = 'F')
        #print('\nMatriz de matrices de correlacion no coherentes formateada:  ', correlation_matrix_non_coherent.shape)

        acquisitions_matrix_state = np.array(acquisitions_matrix_state)                                                           #(sat*n_dwells,)
        acquisitions_matrix_state = acquisitions_matrix_state.reshape(len(sats), acquisitions_matrix_state.shape[0]//len(sats), order = 'F')                   #(sat, n_dwells)
        #print('\nMatriz de Estado de Adquisicion por integracion: (sat, n_integration) ', acquisitions_matrix_state.shape)
        #print(acquisitions_matrix_state)

        acquisition_counter_list, acquisition_sat_state = check_max_dwells_acquistions(acquisitions_matrix_state)

        #print('\nMatriz de Cantidad de Adquisiciones totales: ', np.array(acquisition_counter_list).shape)
        #print(acquisition_counter_list)

        #print('\nMatriz de Estado de Adquisicion no coherente: ', np.array(acquisition_sat_state).shape)
        #print(acquisition_sat_state)

        doppler_freqs = np.array(doppler_freqs)                                                     #(sat*n_dwells,)
        doppler_freqs = doppler_freqs.reshape(len(sats), doppler_freqs.shape[0]//len(sats), order = 'F')          #(sat, n_dwells)
        #print('\nMatriz de Retrasos doppler: ', doppler_freqs.shape)

        delay_codes = np.array(delay_codes)                                                         #(sat*n_dwells,)
        delay_codes = delay_codes.reshape(len(sats), delay_codes.shape[0]//len(sats), order = 'F')                #(sat, n_dwells)
        #print('\nMatriz de Delays: ', delay_codes.shape)

        #Volcado de archivo
        if (dump_file):
                    
            with Pool(cpu_count() - 1) as p:

                p.starmap(dump, [(sv, t, correlation_matrix_non_coherent, doppler_freqs, delay_codes, acquisitions_matrix_state, acquisition_counter_list, acquisition_sat_state) for sv in sats])

    
        print('\n','-'*40)


    #(sat*n_dwells, freq, code)
    correlation_matrix_temporal = np.array(correlation_matrix_temporal)
    #print('\nMatriz de todas las matrices de correlacion sin formatear: (sat*n_dwells, freq, code) ', correlation_matrix_temporal.shape)

    #(sat, integration(n_dwell), freq, code)  --->   Almacena todas las integraciones en una sola matriz para todos los satelites.  -- IMPLEMENTACION ACTUAL PARA GRAFICAR (particularizar para un sat)
    correlation_matrix_temporal = correlation_matrix_temporal.reshape(len(sats), correlation_matrix_temporal.shape[0]//len(sats), correlation_matrix_temporal.shape[1], correlation_matrix_temporal.shape[2], order = 'F')
    #print('\nMatriz de todas las matrices de correlacion formateada: (sat, n_integration, freq, delay) = ', correlation_matrix_temporal.shape)

    #(sat, second, integration(n_dwell), freq, delay) --->   Almacena todas las integraciones en una matriz separando por segundo de analisis e integracion no coherente -- FORMA MAS ORDENADA Y DISCRIMINADA
    correlation_matrix_temporal_2 = correlation_matrix_temporal.reshape(len(sats), analysis_time, number_dwells, correlation_matrix_temporal.shape[2], correlation_matrix_temporal.shape[3])
    #print('\nMatriz de todas las matrices de correlacion formateada: (sat, second, n_integration, freq, delay) = ', correlation_matrix_temporal_2.shape)


    if (graphic and search_option == 2):

        #Surface 3D
        fig = plt.figure("Search Space for Acquisition of Satellite {}".format(satellite))
        ax = Axes3D(fig)
        ax.set_xlabel("Code Phase [chips]",fontweight="bold", fontsize=12)
        ax.set_ylabel("Doppler [Hz]",fontweight="bold", fontsize=12)
        ax.set_yticks([-doppler_max,-doppler_max/2,0,doppler_max/2,doppler_max])        #original
        #ax.set_yticks([doppler_max,doppler_max/2,0,-doppler_max/2,-doppler_max],[doppler_max,doppler_max/2,0,-doppler_max/2,-doppler_max])        #trucada
        ax.set_zlabel("Magnitud",fontweight="bold", fontsize=12)
        ax.set_facecolor('white')
        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        X,Y = np.meshgrid(np.reshape(np.arange(0,1023,1/bin_chip),correlation_matrix_temporal[0][0].shape[1]), range(doppler_max, -doppler_max-1, -doppler_step))  #cambiar por range(doppler_max, -doppler_max-1, -doppler_step) para cambiar signo ejes
        Z = correlation_matrix_temporal[0]
        
        z_max=round(Z.max(),4)
        ax.set_zlim(0, z_max*0.8)
        title = ax.text(0,0,z_max*0.9,"Satellite {} Search Space - Second {} - Integration {}".format(satellite,seconds,0), fontsize=14, horizontalalignment='left', fontweight="bold")

        #init surface 0
        surf = ax.plot_surface(X, Y, Z[0], cmap='jet', rstride=1, cstride=1, linewidth=0, alpha=1, vmin=0, vmax=z_max)     #autumn_r/jet/hot/afmhot
        cb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        #cb.ax.set_title('Correlation Power',fontsize=10,fontweight="bold")
        cb.set_ticks([0, z_max/2, z_max])
        #cb.set_ticklabels([0, Z.max()/2, Z.max()])

        ani = animation.FuncAnimation(fig, update_plot, len(Z), fargs=(Z, X, Y, surf, cb, title, ax), interval=30, blit=False, repeat=False, cache_frame_data=False)


    tiempo_final=time()
    tiempo_ejecucion = tiempo_final - tiempo_inicial
    print('\nSe analizaron: {} s'.format(analysis_time))
    print('\nEl tiempo de ejecucion fue: {} s'.format(round(tiempo_ejecucion,2)))
    print('\n','-'*40)

    plt.show()



if __name__ == "__main__":


    # PARAMETROS DE PRUEBA

    file = '/home/lucas/Tesis/gnss-sdr/prueba1/signal_source.dat'       # APRECIABLES: 2,12,24,29   SNR BAJO: 5,6,25,32
    #file = '/home/lucas/Tesis/rtl-sdr-record/capture1.dat'             # APRECIABLES: 2,12,25,29   SNR BAJO:       SEGUIR BUSCANDO
    format_file = 1                                                     #(gnss-sdr = 1 / rtl_sdr = 2)
    search_option = 2                                                   #(search all satellites = 1 / one satellite = 2)
    satellite = 12                                                      #Satelite a buscar
    sample_rate = 2e6                                                   #Frecuencia de muestreo
    analysis_time = 5                                                   #Tiempo total de analisis en segundos
    coherent_integration_time = 2e-3                                    #Duracion de la ventana de integracion coherente        #minimo 2ms (por problema de rolleo para fs baja) y eso daria step=250    
    number_dwells = int(1)                                              #Numero de integraciones no coherentes a realizar
    dwell_time = number_dwells * coherent_integration_time              #Duracion de la ventana de integracion no coherente
    max_dwells = int(1)                                                 #Numero de integraciones no coherentes que deben dar adquisicion positiva para considerar adquisicion positiva      #verificar q <= a num_dwells 
    doppler_step = int(1/(coherent_integration_time))                   #tamaño bin_frec = 1/T -----> resolucion frec = 1/2T  
    #doppler_step = 500                                                 #Tamaño del bin frecuencial--- La resolucion sera el mínimo error, que sera el tamaño del bin/2  
    doppler_max = int(5e3)                                              #Frec maxima de busqueda en desplazamiento doppler
    doppler_range = range(-doppler_max, doppler_max+1, doppler_step)    #Rango Doppler
    bin_chip = 1                                                        #Resolucion del bin chip
    treshold = 0.019                                                    #Umbral fijo de busqueda
    dump_file = False                                                    #Volcado de resultado en archivos
    graphic = True                                                      #Graficar
    freq_list = np.concatenate((np.arange(0,doppler_max+1,doppler_step),np.arange(-doppler_max,0,doppler_step)),axis=None)

    main()
