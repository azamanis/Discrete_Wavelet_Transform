import numpy as np
import py

def extend_signal(signal, mode):
    return signal

def get_universal_threshold(first_details, N):
    '''
    Obtiene el umbral universal a partir de una estimación
    del nivel de ruido de la señal.
    :param first_details: Detalles de la primera descomposición
    :param N: Número de muestras de la señal
    :return: Umbral universal
    '''
    # MAD(X) = median(|X - median(X)|)
    # sigma = mad(h1)/0.6745 = mad(h1) * 1.48258
    median = np.median(first_details)
    mad = np.median(np.abs(first_details - median))

    scaling_factor = 0.6751
    stimate = mad / scaling_factor

    threshold = stimate * np.sqrt(2 * np.log(N))
    return threshold


def hard_threshold(details, threshold):
        return np.where(np.abs(details) < threshold, 0, details)

def soft_threshold(details, threshold):
        return np.sign(details) * np.where(np.abs(details) < threshold, 0, np.abs(details) - threshold)

def apply_threshold(details, threshold, mode='hard'):
    if mode == 'hard':
        return [hard_threshold(detail, threshold) for detail in details]
    
    if mode == 'soft':
        return [soft_threshold(detail, threshold) for detail in details]

def get_highpass_filter(low_pas_filter):
    high_pass = np.copy(low_pas_filter)
    high_pass = np.flip(high_pass)
    high_pass[::2] *= -1
    return high_pass

def apply_deconstruction(signal, low_pas_filter, mode='zero'):
    '''
    Aplica la transforada de ondículas a una señal
    :param signal: Señal a descomponer
    :param low_pas_filter: Filtro pasa bajos
    :param mode: Modo de extensión de la señal
    :return: Tupla de listas (aproximación, detalles)
    '''

    signal = extend_signal(signal, mode)
    high_pass = get_highpass_filter(low_pas_filter)

    aprox = np.convolve(signal, low_pas_filter)
    details = np.convolve(signal, high_pass)

    return aprox[1::2], details[1::2]

def apply_construction(aprox, details, low_pass_filter):
    '''
    Aplica la inversa transforada inversa de ondículas a una señal
    :param aprox: aproximación
    :param details: Detalles
    :param low_pass_filter: Filtro pasa bajos
    :return: Señal reconstruida
    '''

    l = len(low_pass_filter)
    high_pass_filter = get_highpass_filter(low_pass_filter)

    _low  = np.flip(low_pass_filter)
    _high = np.flip(high_pass_filter)
    
    _aprox = np.zeros(len(aprox)*2)
    _aprox[::2] = aprox

    _details = np.zeros(len(details)*2)
    _details[::2] = details

    r1 = np.convolve(_aprox, _low) 
    r2 = np.convolve(_details, _high)
    
    if len(r1) > len(r2):
        r2 = np.pad(r2, (0, len(r1) - len(r2)), mode='constant')
   
    r = r1 + r2

    return r[l-2:-l+1]


def thresholding(deatails,N, threshold_value, threshold_mode):
    '''
    Función auxiliar para tratar threshold universal
    '''
    if threshold_mode == 'universal':
        threshold_value = get_universal_threshold(deatails[-1], N)
        threshold_mode = 'soft'
        
    return apply_threshold(deatails, mode = threshold_mode, threshold=threshold_value)



def dwt(signal, low_pass_filter, level=1, mode='zero', threshold_value=None, threshold_mode=None):
    ''' 
    Aplica la transforada de ondículas
    :param signal: Señal a descomponer
    :param low_pass_filter: Filtro pasa bajos
    :param level: Nivel de descomposición
    :param mode: Modo de extensión
    :param threshold: Umbral de truncamiento
    :return: Aproximación y detalles a1, [d1, d2, ..., dn]
    '''

    aprox = signal
    details = []
    for _ in range(level):
        aprox, detail = apply_deconstruction(aprox, low_pass_filter, mode)
        details.insert(0, detail)

    if threshold_mode:
        details = thresholding(details,len(signal), threshold_value, threshold_mode )
        
    return aprox, details

def idwt(aprox, details, low_pass_filter, level=None, mode='zero', length=None):
    ''' 
    Aplica la transforada inversa de ondículas
    :param aprox: Aproximación a1
    :param details: Detalles (d1, d2, ..., dn)
    :param low_pass_filter: Filtro de paso bajo
    :param level: Nivel de descomposición
    :return: Señal reconstruida
    '''

    if level is None:
        level = len(details)
    
    signal = aprox
    for detail in details:
        signal = apply_construction(signal, detail, low_pass_filter)

   
    if length:
        signal = signal[:length]

    return signal

def denoise(signal, low_pas_filter, level=1, mode='zero'):
    aprox, details = dwt(signal, low_pas_filter, level, mode, threshold_mode='universal')
    return idwt(aprox, details, low_pas_filter, level, mode, len(signal))


###################### 2 DIMNESIONS ############################

def apply_deconstruct_2d(image, low_pass_filter, mode='zero'):
    '''
    Aplica la transforada de ondículas a una matriz
    :param signal: Señal a descomponer (matriz)
    :param low_pass_filter: Filtro pasa bajos
    :param level: Nivel de descomposición
    :param mode: Modo de extensión de la señal
    :return: cA, cH, cV, cD
    '''
    
    coeffs_rows = []

    # Aplicamos dwt a las filas
    for row in image:
        cA_row, cD_row = apply_deconstruction(row, low_pass_filter, mode=mode)
        coeffs_rows.append(np.concatenate((cA_row, cD_row)))
        

    coeffs_rows = np.array(coeffs_rows)

    
    # Aplicamos dwt a las columnas (del resultado anterior)
    coeffs = []
    for col in coeffs_rows.T:
        cA_col, cD_col = apply_deconstruction(col, low_pass_filter, mode=mode)
        coeffs.append(np.concatenate((cA_col, cD_col)))

    coeffs = np.array(coeffs).T

    #dividimos la matriz en 4 partes
    #  cA | cH
    #  --------
    #  cV | cD
    
    cA = coeffs[:len(coeffs)//2, :len(coeffs)//2]
    cH = coeffs[:len(coeffs)//2, len(coeffs)//2:]
    cV = coeffs[len(coeffs)//2:, :len(coeffs)//2]
    cD = coeffs[len(coeffs)//2:, len(coeffs)//2:]

    return cA, cH, cV, cD
    
    
def dwt_2d(image, low_pass_filter, level=1, mode='zero', threshold_value=None, threshold_mode=None):
    ''' 
    Aplica la transforada de ondículas a una matriz
    :param signal: Señal a descomponer (matriz)
    :param low_pass_filter: Filtro pasa bajos
    :param level: Nivel de descomposición
    :param mode: Modo de extensión de la señal
    :param threshold: Umbral de truncamiento
    :return: Aproximación y detalles cA, [detail1, detail2, ..., detailn] 
    siendo detail_i = [cHi, cVi, cDi]
    '''

    #apply deconstruction to the cA part

    details = []

    for _ in range(level):
        cA, cH, cV, cD = apply_deconstruct_2d(image, low_pass_filter, mode=mode)
        image = cA
        details.insert(0, [cH, cV, cD])
    
    #TODO:
    # if threshold_mode:
    #     details = thresholding(details,len(image), threshold_value, threshold_mode )

    return cA, details

def join_aprox_and_details(aprox, details, casting=False, padding=True):
    '''
    Une la aproximación y los detalles de la descomposición para su presentación
    NO ES UNA RECONSTUCCIÓN
    :param aprox: Aproximación cA
    :param details: lista de detalles [d1,d2,..dn] con d_i=[cH_i, cV_i, cD_i]
    :param casting: realiza casting a uint8
    :param padding: si es True realiza padding de las submatrices menores. 
    En caso contrario, recorta la matriz mayor.
    :return: matrix de coeficientes y detalles
    '''

    n = len(details)

    for i in range(n):
        [cH, cV, cD] = details[i]

        if padding:
        # pad with zeros
            cH = np.pad(cH, (0, len(aprox) - len(cH)), mode='constant')
            cV = np.pad(cV, (0, len(aprox) - len(cV)), mode='constant')
            cD = np.pad(cD, (0, len(aprox) - len(cD)), mode='constant')
        else:
        #recortar matriz de detalles
            n = cH.shape[0]
            aprox = aprox[:n, :n]
        
        aprox = np.concatenate((np.concatenate((aprox, cH), axis=1), np.concatenate((cV, cD), axis=1)), axis=0)
    if casting:
        aprox = aprox.astype(np.uint8)
    return aprox

def idwt_2d(aprox, details, low_pass_filter, level=None, mode='zero', size=None):
    ''' 
    Aplica la transforada inversa de ondículas a una matriz
    :param aprox: Aproximación cA
    :param details: lista de detalles [d1,d2,..dn] con d_i=[cH_i, cV_i, cD_i]
    :param level: Nivel de descomposición
    :param mode: Modo de extensión de la señal
    :return: imagen reconstruida
    '''
    if level is None:
        level = len(details)

    for _ in range(level):
        [cH, cV, cD] = details.pop(0)
        aprox = join_aprox_and_details(aprox, [[cH, cV, cD]])
        col_aprox = []
        row_aprox = []

        #reconstruir las columnas
        for col in aprox.T:
            cA_col = col[:len(col)//2]
            cD_col = col[len(col)//2:]
            _col = apply_construction(cA_col, cD_col, low_pass_filter)
            col_aprox.append(_col)
        col_aprox = np.array(col_aprox).T
        

        for row in col_aprox:
            cA_row = row[:len(row)//2]
            cD_row = row[len(row)//2:]
            _row = apply_construction(cA_row, cD_row, low_pass_filter)
            row_aprox.append(_row)
        row_aprox = np.array(row_aprox)
            
        aprox = row_aprox
    
    if size:
        aprox = aprox[:size,:size]

    return aprox
    
    