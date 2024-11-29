#!/usr/bin/env python3      
#Permite correr directamente con ./script.py si se le da permido chmod +x script.py
# -*- coding: utf-8 -*-

import os 
import subprocess as sp
from time import time
import PCPS

header = '''
------------------ PyGNSS-SDR ------------------

Bloque de Adquisicion para señales GPS L1/CA   
Laboratorio de Telecomunicaciones - FACET - UNT
CASTILLO DELACROIX LUCAS    -    8/2021
'''
#Algoritmo: Parallel Code Phase Search 

# Menu Principal
def menu():
    """
    Función que limpia la pantalla y muestra nuevamente el menu
    """
    os.system('clear')
    print(header)
    print('-'*48)
    print("\nMenu Principal\n")
    print("\t1 - Analizar un archivo grabado")
    print("\t2 - Grabar un archivo")
    print("\t3 - Salir\n")
    print('-'*48)

    return input("\nIngrese una opcion: ")

# Analizar un archivo grabado
def submenu_1():

    os.system('clear')
    print("Selecciona una opción\n")
    print("\t1 - Correr Parametros por Default")
    print("\t2 - Setear Parametros")
    print("\t3 - Volver al menu anterior")
    print("\t4 - Salir")

    return input("\nIngrese una opcion: ")

def submenu_1_2():

    os.system('clear')
    print("Selecciona una opción\n")
    print("\t1 - Volver a setear parametros")
    print("\t2 - Volver al menu principal")

    return input("\nIngrese una opcion: ")

# Input Required Parameters (option 1 and option 2)
def set_req_parameters():

    os.system('clear')
    print('\nIngresar Parametros')
    print('\n','-'*40)

    PCPS.graphic = False
    PCPS.dump_file = False 
        
    # Input File and check exist
    while(True):
        
        PCPS.file = input("\nIngrese la ruta del archivo: ")                                                 #'/home/lucas/Tesis/gnss-sdr/prueba1/signal_source.dat' 
        
        if(os.path.isfile(PCPS.file)):
            break

        else:
            print("\nEl archivo no existe... Intente de nuevo")
            continue

    # Format File option and check
    while(True):
        
        PCPS.format_file = int(input("\nSeleccione el formato del archivo: \n\n\t1 - GNSS-SDR \n\t2 - RTL-SDR \n\nIngrese una Opcion: "))       #(gnss-sdr = 1 / rtl_sdr = 2)
        
        if (PCPS.format_file == 1 or PCPS.format_file == 2):
            break
        else:
            print("\nOpcion invalida... Intente de nuevo")
            continue

    # Input Sample rate of record file
    PCPS.sample_rate = float(input("\nIngrese la frecuencia de muestro del archivo [Hz]: "))                                    #Frecuencia de muestreo                                                                                     

    # Input Search Option and check
    while(True):
        
        PCPS.search_option = int(input("\nSeleccione el tipo de busqueda: \n\n\t1 - Todos los satelites \n\t2 - Un satelite particular \n\nIngrese una Opcion: "))     #(search all PCPS.satellites = 1 / one PCPS.satellite = 2)
        
        if (PCPS.search_option == 1 or PCPS.search_option == 2):
            break
        else:
            print("\nOpcion invalida... Intente de nuevo")
            continue

    # Input Satellite number and check    
    if PCPS.search_option == 2: 

        while(True):
        
            PCPS.satellite = int(input("\nIngrese el satelite que desea buscar: "))   
            
            if (PCPS.satellite in range(1,33,1)):
                break
            else:
                print("\nNumero de Satelite invalido... Intente de nuevo")
                continue

        if(yes_or_no('\nDesea graficar(y/n): ')):
            PCPS.graphic = True

    # Almacenar resultados en archivo
    if(yes_or_no('\nDesea grabar los resultados en un archivo(y/n): ')):
        PCPS.dump_file = True

    # Input Analysis Time Option and check
    while(True):
        
        PCPS.analysis_time = int(input("\nIngrese el tiempo que desea analizar el archivo [seg]: "))              #Tiempo total de analisis en segundos
        
        if (PCPS.analysis_time > 0):
            break
        else:
            print("\nDebe ingresar un tiempo mayor a 0 ... Intente de nuevo")
            continue

# Set Default Parameters
def default_parameters():

    set_req_parameters()
    PCPS.coherent_integration_time = 2e-3                                                   #Duracion de la ventana de integracion coherente        #minimo 2ms (por problema de rolleo para fs baja) y eso daria step=250    
    PCPS.number_dwells = int(1)                                                             #Numero de integraciones no coherentes a realizar
    PCPS.dwell_time = PCPS.number_dwells * PCPS.coherent_integration_time                   #Duracion de la ventana de integracion no coherente
    PCPS.max_dwells = int(1)                                                                #Numero de integraciones no coherentes que deben dar adquisicion positiva para considerar adquisicion positiva  
    #PCPS.doppler_step = 500                                                                 #Resolucion bin frecuencial   (resolucion minima ideal = 500)               #PCPS.doppler_step = int(1/(2*PCPS.coherent_integration_time))   #bin_frec = 1/2T o 2/3T
    PCPS.doppler_step = int(1/PCPS.coherent_integration_time)
    PCPS.doppler_max = int(5e3)                                                             #Frec maxima de busqueda en desplazamiento doppler
    PCPS.doppler_range = range(-PCPS.doppler_max, PCPS.doppler_max+1, PCPS.doppler_step)    #Rango Doppler
    PCPS.bin_chip = 1                                                                       #Resolucion del bin chip
    PCPS.treshold = 0.019                                                                   #Umbral fijo de busqueda

# Set All parameters
def set_parameters():

    set_req_parameters()
    PCPS.coherent_integration_time = float(input("\nIngrese el tiempo de integracion coherente [seg]: "))                            #Duracion de la ventana de integracion coherente        #minimo 2ms (por problema de rolleo para fs baja) y eso daria step=250    
    PCPS.number_dwells = int(input("\nIngrese el numero de integraciones coherentes por cada ventana no coherente: "))               #Numero de integraciones no coherentes a realizar                             
    PCPS.dwell_time = PCPS.number_dwells * PCPS.coherent_integration_time                                                                      #Duracion de la ventana de integracion no coherente
    
    # Input max_dwells and check <= num_dwells
    while(True):
        
        PCPS.max_dwells = int(input("\nIngrese el numero de integraciones en la ventana no coherente que deben ser positivas para lograr adquisicion: "))        #Numero de integraciones no coherentes que deben dar adquisicion positiva para considerar adquisicion positiva
        
        if (PCPS.max_dwells <= PCPS.number_dwells):
            break
        else:
            print("\nEl numero de integraciones positivas para adquisicion positiva, no puede ser mayor que el numero de integraciones realizadas")
            print("\nIntente de nuevo")
            continue   
    
    #PCPS.doppler_step = int(input("\nIngrese la resolucion del bin Frecuencial: \n\n\tRecomendado: 500 o 250 \n\nIngrese un valor: "))                                              #Resolucion bin frecuencial   (resolucion minima ideal = 500)       #PCPS.doppler_step = int(1/(2*PCPS.coherent_integration_time))      #bin_frec = 1/2T o 2/3T  
    PCPS.doppler_step = int(1/(PCPS.coherent_integration_time))
    PCPS.doppler_max = int(float(input("\nIngrese la Frecuencia maxima de busqueda para el desplazamiento doppler [Hz]: \n\n\tRecomendado \n\n\tReceptores dinamicos: 10e3  \n\tReceptores estaticos: 5e3 \n\nIngrese un valor: ")))          #Frec maxima de busqueda en desplazamiento doppler (5e3 estaticos - 10e3 dinamicos)                                                                                        
    PCPS.doppler_range = range(-PCPS.doppler_max, PCPS.doppler_max+1, PCPS.doppler_step)                                                                                                                            #Rango Doppler
    
    # Input bin_chip resolution Option and check
    # while(True):
        
    #     PCPS.bin_chip = int(input("\nSeleccione la resolucion del bin chip: \n\n\t1 - Cada 1 chip de codigo \n\t2 - Cada 0.5 chip de codigo \n\nIngrese una Opcion: "))                 #Resolucion del bin chip        #PCPS.bin_chip = 1 o 2 
        
    #     if (PCPS.bin_chip == 1 or PCPS.bin_chip == 2):
    #         break
    #     else:
    #         print("\nOpcion invalida... Intente de nuevo")
    #         continue

    PCPS.bin_chip = 1
    PCPS.treshold = float(input("\nIngrese el valor del umbral de deteccion: "))                                                     #Umbral fijo de busqueda        #PCPS.treshold = 0.005

# Show Inputs Parameters
def show_parameters():

    os.system('clear')
    print("Los parametros ingresados son: \n")
    print('-'*40)
    
    print("\nRuta del archivo: ", PCPS.file)

    if(PCPS.format_file == 1): 
        print("\nFormato del archivo: GNSS-SDR")

    elif(PCPS.format_file == 2): 
        print("\nFormato del archivo: RTL-SDR")

    if(PCPS.search_option == 1): 
        print("\nBuscar todos los satelites")

    elif(PCPS.search_option == 2): 
        print("\nBuscar el satelite: ", PCPS.satellite)
        print("\nGraficar: ", PCPS.graphic)
    
    print("\nFrecuencia de Muestreo: {} Hz".format(PCPS.sample_rate))
    print("\nTiempo total de analisis: {} s".format(PCPS.analysis_time))
    print("\nTiempo de integracion coherente: {} s".format(PCPS.coherent_integration_time))
    print("\nNumero de integraciones por cada ventana no coherente: {}".format(PCPS.number_dwells))
    print("\nTiempo total de la ventana de integracion no coherente: {} s".format(PCPS.dwell_time))
    print("\nNumero de adquisiciones positivas por cada ventana no coherente para lograr adquisicion: {}".format(PCPS.max_dwells))
    print('\nRango de busqueda Frecuencial: ± {} Hz'.format(PCPS.doppler_max))
    print("\nResolución del bin frecuencial: {} Hz".format(PCPS.doppler_step))
    #print("\nResolucion del bin frecuencial: ± {} Hz ??".format(PCPS.doppler_step/2))
    #print('\nResolucion del bin chip: {}'.format(1/PCPS.bin_chip))
    print('\nUmbral de deteccion: {}'.format(PCPS.treshold))
    print("\nVolcar resultados en archivo: ", PCPS.dump_file)
    print('\n','-'*40)

# Grabar un archivo
def submenu_2():

    os.system('clear')
    print("Selecciona una opción\n")
    print("\t1 - Setear Parametros de grabacion y Grabar")
    print("\t2 - Volver al menu anterior")
    print("\t3 - Salir")

    return input("\nIngrese una opcion: ")

# Setear Parametros de grabacion y Grabar
def rtl_sdr_record():
        
    record_file_name = input("\nIngrese el nombre del archivo que se grabara (sin extension): ") + '.dat'
    sample_rate = float(input("\nIngrese la frecuencia de muestro del archivo: "))                #2e6
    gain = float(input("\nIngrese la ganancia del front-end RF: "))                               #40
    second = int(input("\nIngrese la cantidad de segundos a grabar: "))
    number_samples = second * sample_rate
    f_carrier_gps = 1575420000

    sp.run(['rtl_biast','-b','1'])
    sp.run(['rtl_sdr',record_file_name,'-s',str(sample_rate),'-f',str(f_carrier_gps),'-g',str(gain),'-n',str(number_samples)])
    #sp.run(['rtl_sdr','capture2.dat','-s','2e6','-f','1575420000','-g','40','-n','20000000'])
    input("\nPulse una tecla para volver al menu principal")

# Funcion auxiliar Pedir confirmacion
def yes_or_no(question):
    
    reply = str(input(question)).lower().strip()
    
    if reply[0] == 'y':
        return 1
    elif reply[0] == 'n':
        return 0
    else:
        return yes_or_no("\nPlease Enter (y/n) ")




#MAIN MENU
def main():

    while (True):

        option = menu()
        
        # Analizar un archivo grabado
        if option == "1":
            
            option_submenu1 = submenu_1()

            # Correr Parametros por Default
            if option_submenu1 == "1":
                
                while(True):

                    default_parameters()
                    show_parameters()

                    if(yes_or_no('\nConfirmar parametros(y/n): ')):

                        #Correr algoritmo
                        os.system('clear')
                        print('\n','-'*40)
                        print("\nProcesando archivo...")

                        PCPS.main()
                        
                        print("\nAnalisis Finalizado\n")
                        print('\n','-'*40)
                        input("\nPulse una tecla para volver al menu principal")

                        break
                    
                    else:
                        
                        option_submenu1_2 = submenu_1_2()
                        
                        # Volver a setear Parametros
                        if option_submenu1_2 == "1":
                            continue

                        # Volver al menu principal
                        elif option_submenu1_2 == "2":
                            break
            
            # Setear Parametros
            elif option_submenu1 == "2":
                
                while(True):

                    set_parameters()
                    show_parameters()

                    if(yes_or_no('\nConfirmar parametros(y/n): ')):

                        #Correr algoritmo
                        os.system('clear')
                        print('\n','-'*40)
                        print("\nProcesando archivo...")
                        
                        PCPS.main()
                        
                        print("\nAnalisis Finalizado\n")
                        print('\n','-'*40)
                        input("\nPulse una tecla para volver al menu principal")

                        break

                    else:
                        
                        option_submenu1_2 = submenu_1_2()
                        
                        # Volver a setear Parametros
                        if option_submenu1_2 == "1":
                            continue

                        # Volver al menu principal
                        elif option_submenu1_2 == "2":
                            break

            # Volver al menu anterior
            elif option_submenu1 == "3":
                continue
            
            # Salir del programa
            elif option_submenu1 == "4":
                os.system('clear')
                break

            # Opcion invalida
            else:
                input("\nOpcion invalida... pulsa una tecla para volver al menu principal")
            
        # Setear Parametros de grabacion y Grabar
        elif option == "2":
            
            option_submenu2 = submenu_2()

            # Setear Parametros de grabacion
            if option_submenu2 == "1":
                rtl_sdr_record()
            
            # Volver al menu anterior
            elif option_submenu2 == "2":
                continue
            
            # Salir del programa
            elif option_submenu2 == "3":
                os.system('clear')
                break

            # Opcion invalida
            else:
                input("\nOpcion invalida... pulsa una tecla para volver al menu principal")

        # Salir del programa
        elif option == "3":
            os.system('clear')
            break

        # Opcion invalida
        else:
            input("\nOpcion invalida... pulsa una tecla para volver a intentar")
       


if __name__ == "__main__":
    main()