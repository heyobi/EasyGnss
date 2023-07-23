
def main_gps_calc(Obs_path,sp3_path):
    import numpy as np
    from numpy import mat
    import gnsspy as gp

    gozlem = gp.read_obsFile(Obs_path) #ista0070.20o
    uydular = gp.read_sp3File(sp3_path) #igs20872.sp3


    # Receiver's Approximate Coordinates
    G_x = gozlem.approx_position[0]
    G_y = gozlem.approx_position[1]
    G_z = gozlem.approx_position[2]

    # Edited the code so that any observation and sp3 file can be read
    tarih = str(gozlem.observation.index[0][0])
    tarih2 = ""
    for i in tarih:
        if i == " ":
            break
        else:
            tarih2 +=i

 


    # Gps satellite names for each code read from observation file
    def saath(saat):
        uydulist0 = []
        for i in gozlem.observation.index:
            if str(i[0]) == tarih2+" "+saat:
                uydulist0.append(i[1])
        uyduset0 = set()
        for i in uydulist0:
            for j in i:
                if j == "G":
                    uyduset0.add(i)
        m = tarih2 + " " + saat
        return sorted(set(uyduset0),key=uydulist0.index),uydular[m:m]

    uyduset0,uydu0 = saath("00:00:00")
    uyduset6,uydu6 = saath("06:00:00")
    uyduset12,uydu12 = saath("12:00:00")
    uyduset18,uydu18 = saath("18:00:00")
    # Getting Satellite Coordinates from sp3

    # function that retrieves satellite coordinates for every hour
    def saat_koord(uydu,uyduset):
        
        #getting x values
        uydu_x = np.zeros((len(uyduset),1))
        uydu=uydu.filter(items=['Sv',"X","Y","Z","deltaT"])
        say = 0
        for i in uyduset:
            uydu_x[say][0] = float(uydu.filter(like=i, axis=0).X)
            say +=1

        #getting y values
        uydu_y = np.zeros((len(uyduset),1))
        uydu=uydu.filter(items=['Sv',"X","Y","Z","deltaT"])
        say = 0
        for i in uyduset:
            uydu_y[say][0] = float(uydu.filter(like=i, axis=0).Y)
            say +=1

        #getting z values
        uydu_z = np.zeros((len(uyduset),1))
        uydu=uydu.filter(items=['Sv',"X","Y","Z","deltaT"])
        say = 0
        for i in uyduset:
            uydu_z[say][0] = float(uydu.filter(like=i, axis=0).Z)
            say +=1

        #getting deltaT values
        uydu_d = np.zeros((len(uyduset),1))
        uydu=uydu.filter(items=['Sv',"X","Y","Z","deltaT"])
        say = 0
        for i in uyduset:
            uydu_d[say][0] = float(uydu.filter(like=i, axis=0).deltaT)
            say +=1
        
        #satellite units are converted to meters, clock error is calculated and its unit is sec
        uydu_x,uydu_y,uydu_z,uydu_d = uydu_x*1000,uydu_y*1000,uydu_z*1000,uydu_d*10**(-6)*299792458
        return uydu_x,uydu_y,uydu_z,uydu_d


    #satellite coordinates (m), dts(s)
    uydu0_x,uydu0_y,uydu0_z,uydu0_d = saat_koord(uydu0,uyduset0)
    uydu6_x,uydu6_y,uydu6_z,uydu6_d = saat_koord(uydu6,uyduset6)
    uydu12_x,uydu12_y,uydu12_z,uydu12_d = saat_koord(uydu12,uyduset12)
    uydu18_x,uydu18_y,uydu18_z,uydu18_d = saat_koord(uydu18,uyduset18)
    uydu0_t = uydu0_x,uydu0_y,uydu0_z,uydu0_d
    uydu6_t = uydu6_x,uydu6_y,uydu6_z,uydu6_d
    uydu12_t = uydu12_x,uydu12_y,uydu12_z,uydu12_d
    uydu18_t = uydu18_x,uydu18_y,uydu18_z,uydu18_d




 
    # Getting C1 and P2 Codes in observation file. (There is no P1 code)
    C1_0 = gozlem.observation["C1"].filter(like=tarih2+" " + "00:00:00", axis=0)[0:len(uyduset0)]
    C1_6 = gozlem.observation["C1"].filter(like=tarih2+" " + "06:00:00", axis=0)[0:len(uyduset6)]
    C1_12 = gozlem.observation["C1"].filter(like=tarih2+" " + "12:00:00", axis=0)[0:len(uyduset12)]
    C1_18 = gozlem.observation["C1"].filter(like=tarih2+" " + "18:00:00", axis=0)[0:len(uyduset18)]

    P2_0 = gozlem.observation["P2"].filter(like=tarih2+" " + "00:00:00", axis=0)[0:len(uyduset0)]
    P2_6 = gozlem.observation["P2"].filter(like=tarih2+" " + "06:00:00", axis=0)[0:len(uyduset6)]
    P2_12 = gozlem.observation["P2"].filter(like=tarih2+" " + "12:00:00", axis=0)[0:len(uyduset12)]
    P2_18 = gozlem.observation["P2"].filter(like=tarih2+" " + "18:00:00", axis=0)[0:len(uyduset18)]

 
    #Getting C1 codes for hours
    C1_0n = np.array(C1_0)
    C1_0n.shape = (len(C1_0n),1)
    C1_6n = np.array(C1_6)
    C1_6n.shape = (len(C1_6n),1)
    C1_12n = np.array(C1_12)
    C1_12n.shape = (len(C1_12n),1)
    C1_18n = np.array(C1_18)
    C1_18n.shape = (len(C1_18n),1)

    #Getting P2 codes for hours
    P2_0n = np.array(P2_0)
    P2_0n.shape = (len(P2_0n),1)
    P2_6n = np.array(P2_6)
    P2_6n.shape = (len(P2_6n),1)
    P2_12n = np.array(P2_12)
    P2_12n.shape = (len(P2_12n),1)
    P2_18n = np.array(P2_18)
    P2_18n.shape = (len(P2_18n),1)


    # Calculation of satellite-receiver distances

    #q values are equal for c1 and p2
    #Distance between satellite and observation point
    q0 = np.zeros((len(uydu0_x),1))
    for i in range(len(uydu0_x)):
        q0[i][0] = ((G_x-uydu0_x[i])**2 + (G_y-uydu0_y[i])**2 + (G_z-uydu0_z[i])**2)**0.5

    q6 = np.zeros((len(uydu6_x),1))
    for i in range(len(uydu6_x)):
        q6[i][0] = ((G_x-uydu6_x[i])**2 + (G_y-uydu6_y[i])**2 + (G_z-uydu6_z[i])**2)**0.5

    q12 = np.zeros((len(uydu12_x),1))
    for i in range(len(uydu12_x)):
        q12[i][0] = ((G_x-uydu12_x[i])**2 + (G_y-uydu12_y[i])**2 + (G_z-uydu12_z[i])**2)**0.5

    q18 = np.zeros((len(uydu18_x),1))
    for i in range(len(uydu18_x)):
        q18[i][0] = ((G_x-uydu18_x[i])**2 + (G_y-uydu18_y[i])**2 + (G_z-uydu18_z[i])**2)**0.5


 
    # Calculation of Corrected codes

    #Corrected dimensions of the C1 code
    C1_D_0 = C1_0n+uydu0_d
    C1_D_6 = C1_6n+uydu6_d
    C1_D_12 = C1_12n+uydu12_d
    C1_D_18 = C1_18n+uydu18_d

    #Corrected dimensions of the P2 code
    P2_D_0 = P2_0n+uydu0_d
    P2_D_6 = P2_6n+uydu6_d
    P2_D_12 = P2_12n+uydu12_d
    P2_D_18 = P2_18n+uydu18_d


    #Calculation Of A coefficients matrix
    def A_matrix(uydu_t,q):
        uzunluk = len(uydu_t[0])
        ax = np.zeros((uzunluk,1))
        ay = np.zeros((uzunluk,1))
        az = np.zeros((uzunluk,1))
        ac = np.zeros((uzunluk,1))

        for i in range(uzunluk):
            ax[i] = (G_x-float(uydu_t[0][i]))/q[i]
        for i in range(uzunluk):
            ay[i] = (G_y-float(uydu_t[1][i]))/q[i]
        for i in range(uzunluk):
            az[i] = (G_z-float(uydu_t[2][i]))/q[i]
        for i in range(uzunluk):
            ac[i] = 2997924589
        return np.column_stack((ax, ay,az,ac))

 
    A0 = A_matrix(uydu0_t,q0)
    A6 = A_matrix(uydu6_t,q6)
    A12 = A_matrix(uydu12_t,q12)
    A18 = A_matrix(uydu18_t,q18)

    # N=(A^T*A) Normal Equations Coefficients Matrix
    #Normal denklemler katsayılar matrisi
    N0 = np.dot(mat(np.transpose(A0)),mat(A0))
    N6 = np.dot(mat(np.transpose(A6)),mat(A6))
    N12 = np.dot(mat(np.transpose(A12)),mat(A12))
    N18 = np.dot(mat(np.transpose(A18)),mat(A18))

    #bilinmeyenlerin kofaktör matrisi
    N_inv0 = np.linalg.inv(N0)
    N_inv6 = np.linalg.inv(N6)
    N_inv12 = np.linalg.inv(N12)
    N_inv18 = np.linalg.inv(N18)

 


 
    # Minimized Mesaurments Vectors
    # l Küçültülmüş Ölçüler Vektörü
    C1_l0 = C1_D_0 - q0
    C1_l6 = C1_D_6 - q6
    C1_l12 = C1_D_12 - q12
    C1_l18 = C1_D_18 - q18

    # l Küçültülmüş Ölçüler Vektörü
    P2_l0 = P2_D_0 - q0
    P2_l6 = P2_D_6 - q6
    P2_l12 = P2_D_12 - q12
    P2_l18 = P2_D_18 - q18


    # n = A^T * L 
    C1_n0 = mat(np.transpose(A0))*C1_l0
    C1_n6  = mat(np.transpose(A6))*C1_l6
    C1_n12 = mat(np.transpose(A12))*C1_l12
    C1_n18 = mat(np.transpose(A18))*C1_l18

    P2_n0 = mat(np.transpose(A0))*P2_l0
    P2_n6 = mat(np.transpose(A6))*P2_l6
    P2_n12 = mat(np.transpose(A12))*P2_l12
    P2_n18 = mat(np.transpose(A18))*P2_l18


    # Minimized Unknowns Vector

    #elements of the matrix:
    #δx
    #δy
    #δz
    #δdtr

    #for C1
    C1_x0 = N_inv0*C1_n0
    C1_x6 = N_inv6*C1_n6
    C1_x12 = N_inv12*C1_n12
    C1_x18 = N_inv18*C1_n18

    #For P2
    P2_x0 = N_inv0*P2_n0
    P2_x6 = N_inv6*P2_n6
    P2_x12 = N_inv12*P2_n12
    P2_x18 = N_inv18*P2_n18

 
    # balance Vector
    # V = Ax - l
    # balance vector for C1 codes (V)
    C1_V0 = A0*C1_x0 - C1_l0
    C1_V6 = A6*C1_x6 - C1_l6
    C1_V12 = A12*C1_x12 - C1_l12
    C1_V18 = A18*C1_x18 - C1_l18

    # balance vector for C1 codes (V)
    P2_V0 = A0*P2_x0 - P2_l0
    P2_V6 = A6*P2_x6 - P2_l6
    P2_V12 = A12*P2_x12 - P2_l12
    P2_V18 = A18*P2_x18 - P2_l18

 
    #  Sum of Correction Squares (V^T*V)
    #Correction Sum of Squares (V^T*V) for C1
    C1_Dkt0 = float(np.transpose(C1_V0)*C1_V0)
    C1_Dkt6 = float(np.transpose(C1_V6)*C1_V6)
    C1_Dkt12 = float(np.transpose(C1_V12)*C1_V12)
    C1_Dkt18 = float(np.transpose(C1_V18)*C1_V18)

    #Correction Sum of Squares (V^T*V) for P2
    P2_Dkt0 = float(np.transpose(P2_V0)*P2_V0)
    P2_Dkt6 = float(np.transpose(P2_V6)*P2_V6)
    P2_Dkt12 = float(np.transpose(P2_V12)*P2_V12)
    P2_Dkt18 = float(np.transpose(P2_V18)*P2_V18)

 
    #Level Of Freedom (DoF)
    f0 = len(uyduset0) - 4
    f6 = len(uyduset6) - 4
    f12 = len(uyduset12) - 4
    f18 = len(uyduset18) - 4

 
    #Unit Weighted Measure standard deviation
    C1_S0_0 = (C1_Dkt0/f0)**0.5
    C1_S0_6 = (C1_Dkt6/f6)**0.5
    C1_S0_12 = (C1_Dkt12/f12)**0.5
    C1_S0_18 = (C1_Dkt18/f18)**0.5

    P2_S0_0 = (P2_Dkt0/f0)**0.5
    P2_S0_6 = (P2_Dkt6/f6)**0.5
    P2_S0_12 = (P2_Dkt12/f12)**0.5
    P2_S0_18 = (P2_Dkt18/f18)**0.5


    # Weight coefficient of Adjusted Measure

    # Weight coefficient of balanced Measure
    Qx1x1_0 = float(N_inv0[0].reshape(4,1)[0][0])
    Qy1y1_0 = float(N_inv0[1].reshape(4,1)[1][0])
    Qz1z1_0 = float(N_inv0[2].reshape(4,1)[2][0])

    Qx1x1_6 = float(N_inv6[0].reshape(4,1)[0][0])
    Qy1y1_6 = float(N_inv6[1].reshape(4,1)[1][0])
    Qz1z1_6 = float(N_inv6[2].reshape(4,1)[2][0])

    Qx1x1_12 = float(N_inv12[0].reshape(4,1)[0][0])
    Qy1y1_12 = float(N_inv12[1].reshape(4,1)[1][0])
    Qz1z1_12 = float(N_inv12[2].reshape(4,1)[2][0])

    Qx1x1_18 = float(N_inv18[0].reshape(4,1)[0][0])
    Qy1y1_18 = float(N_inv18[1].reshape(4,1)[1][0])
    Qz1z1_18 = float(N_inv18[2].reshape(4,1)[2][0])

 

    # Standard deviations of unknowns

    # Standard deviations of unknowns
    # C1
    C1_Sx_0 = C1_S0_0*(Qx1x1_0)**0.5
    C1_Sy_0 = C1_S0_0*(Qy1y1_0)**0.5
    C1_Sz_0 = C1_S0_0*(Qz1z1_0)**0.5

    C1_Sx_6 = C1_S0_6*(Qx1x1_6)**0.5
    C1_Sy_6 = C1_S0_6*(Qy1y1_6)**0.5
    C1_Sz_6 = C1_S0_6*(Qz1z1_6)**0.5

    C1_Sx_12 = C1_S0_12*(Qx1x1_12)**0.5
    C1_Sy_12 = C1_S0_12*(Qy1y1_12)**0.5
    C1_Sz_12 = C1_S0_12*(Qz1z1_12)**0.5

    C1_Sx_18 = C1_S0_18*(Qx1x1_18)**0.5
    C1_Sy_18 = C1_S0_18*(Qy1y1_18)**0.5
    C1_Sz_18 = C1_S0_18*(Qz1z1_18)**0.5

    # P2
    P2_Sx_0 = P2_S0_0*(Qx1x1_0)**0.5
    P2_Sy_0 = P2_S0_0*(Qy1y1_0)**0.5
    P2_Sz_0 = P2_S0_0*(Qz1z1_0)**0.5

    P2_Sx_6 = P2_S0_6*(Qx1x1_6)**0.5
    P2_Sy_6 = P2_S0_6*(Qy1y1_6)**0.5
    P2_Sz_6 = P2_S0_6*(Qz1z1_6)**0.5

    P2_Sx_12 = P2_S0_12*(Qx1x1_12)**0.5
    P2_Sy_12 = P2_S0_12*(Qy1y1_12)**0.5
    P2_Sz_12 = P2_S0_12*(Qz1z1_12)**0.5

    P2_Sx_18 = P2_S0_18*(Qx1x1_18)**0.5
    P2_Sy_18 = P2_S0_18*(Qy1y1_18)**0.5
    P2_Sz_18 = P2_S0_18*(Qz1z1_18)**0.5

 

 
    # Positional Standard Deviation

 
    #positional Standard Deviation
    # C1
    C1_psd_0 = (C1_Sx_0**2 + C1_Sy_0**2 + C1_Sz_0**2)**0.5
    C1_psd_6 = (C1_Sx_6**2 + C1_Sy_6**2 + C1_Sz_6**2)**0.5
    C1_psd_12 = (C1_Sx_12**2 + C1_Sy_12**2 + C1_Sz_12**2)**0.5
    C1_psd_18 = (C1_Sx_18**2 + C1_Sy_18**2 + C1_Sz_18**2)**0.5

    # P2
    P2_psd_0 = (P2_Sx_0**2 + P2_Sy_0**2 + P2_Sz_0**2)**0.5
    P2_psd_6 = (P2_Sx_6**2 + P2_Sy_6**2 + P2_Sz_6**2)**0.5
    P2_psd_12 = (P2_Sx_12**2 + P2_Sy_12**2 + P2_Sz_12**2)**0.5
    P2_psd_18 = (P2_Sx_18**2 + P2_Sy_18**2 + P2_Sz_18**2)**0.5

 


 
    # Position Dilution of Precision

 
    PDOP_0 = (Qx1x1_0 + Qy1y1_0 + Qz1z1_0)**0.5
    PDOP_6 = (Qx1x1_6 + Qy1y1_6 + Qz1z1_6)**0.5
    PDOP_12 = (Qx1x1_12 + Qy1y1_12 + Qz1z1_12)**0.5
    PDOP_18 = (Qx1x1_18 + Qy1y1_18 + Qz1z1_18)**0.5

 


 
    # Balanced Reciver Coordinates

    # Balanced reciver Coordinates for C1
    C1_AR_X_0 = float(C1_x0[0]) + G_x
    C1_AR_Y_0 = float(C1_x0[1]) + G_y
    C1_AR_Z_0 = float(C1_x0[2]) + G_z

    C1_AR_X_6 = float(C1_x6[0]) + G_x
    C1_AR_Y_6 = float(C1_x6[1]) + G_y
    C1_AR_Z_6 = float(C1_x6[2]) + G_z

    C1_AR_X_12 = float(C1_x12[0]) + G_x
    C1_AR_Y_12 = float(C1_x12[1]) + G_y
    C1_AR_Z_12 = float(C1_x12[2]) + G_z

    C1_AR_X_18 = float(C1_x18[0]) + G_x
    C1_AR_Y_18 = float(C1_x18[1]) + G_y
    C1_AR_Z_18 = float(C1_x18[2]) + G_z

    # Balanced reciver Coordinates for P2
    P2_AR_X_0 = float(P2_x0[0]) + G_x
    P2_AR_Y_0 = float(P2_x0[1]) + G_y
    P2_AR_Z_0 = float(P2_x0[2]) + G_z

    P2_AR_X_6 = float(P2_x6[0]) + G_x
    P2_AR_Y_6 = float(P2_x6[1]) + G_y
    P2_AR_Z_6 = float(P2_x6[2]) + G_z

    P2_AR_X_12 = float(P2_x12[0]) + G_x
    P2_AR_Y_12 = float(P2_x12[1]) + G_y
    P2_AR_Z_12 = float(P2_x12[2]) + G_z

    P2_AR_X_18 = float(P2_x18[0]) + G_x
    P2_AR_Y_18 = float(P2_x18[1]) + G_y
    P2_AR_Z_18 = float(P2_x18[2]) + G_z



 
    # #Calculation for C1 Code
    #Calculation of A cofficent matrix for all epocs of C1
    def A_matrix(uydu_t,q):
        uzunluk = len(uydu_t[0])
        ax = np.zeros((uzunluk,1))
        ay = np.zeros((uzunluk,1))
        az = np.zeros((uzunluk,1))

        for i in range(uzunluk):
            ax[i] = (G_x-float(uydu_t[0][i]))/q[i]
        for i in range(uzunluk):
            ay[i] = (G_y-float(uydu_t[1][i]))/q[i]
        for i in range(uzunluk):
            az[i] = (G_z-float(uydu_t[2][i]))/q[i]
        return np.column_stack((ax, ay,az))

    A0 = A_matrix(uydu0_t,q0)
    A6 = A_matrix(uydu6_t,q6)
    A12 = A_matrix(uydu12_t,q12)
    A18 = A_matrix(uydu18_t,q18)

 
    SA0 = np.zeros((len(A0),1))
    SA6 = np.zeros((len(A6),1))
    SA12 = np.zeros((len(A12),1))
    SA18 = np.zeros((len(A18),1))

    CA0 = np.ones((len(A0),1))*299792458
    CA6 = np.ones((len(A6),1))*299792458
    CA12 = np.ones((len(A12),1))*299792458
    CA18 = np.ones((len(A18),1))*299792458

 
    #Big A cofficent calculated
    Abig1 = np.column_stack((A0,CA0,SA0,SA0,SA0))
    Abig2 = np.column_stack((A6,SA6,CA6,SA6,SA6))
    Abig3 = np.column_stack((A12,SA12,SA12,CA12,SA12))
    Abig4 = np.column_stack((A18,SA18,SA18,SA18,CA18))
    Abig = np.row_stack((Abig1,Abig2,Abig3,Abig4))
    # A Coffiectn matrix is same for C1 and P2


 
    # Big L Matrix

 
    #Big L matrix for C1
    l_big_C1 = np.row_stack((C1_l0,C1_l6,C1_l12,C1_l18))
    l_big_C1.shape #(doğru birleştimi kontrol)

 
    #Big L matrix for P2
    l_big_P2 = np.row_stack((P2_l0,P2_l6,P2_l12,P2_l18))
    l_big_P2.shape #(doğru birleştimi kontrol

 
    #n for C1
    # n = A^t * L
    n_big_C1 = np.mat(np.transpose(Abig))*l_big_C1

    #n for P2
    n_big_P2 = np.mat(np.transpose(Abig))*l_big_P2


 
    #Same for C1 and P2
    N_big = np.mat(np.transpose(Abig))*Abig

 
    N_big_inv = np.linalg.inv(N_big)

 
    # x = N^-1 * n Minimized Unknowns vector
    #x for C1
    x_big_C1 = N_big_inv * n_big_C1

    #x for P2
    x_big_P2 = N_big_inv * n_big_P2


    # correction vector
    #V for C1
    V_big_C1 = np.mat(Abig)*x_big_C1 - l_big_C1

    #V for C1
    V_big_P2 = np.mat(Abig)*x_big_P2 - l_big_P2


    #corredtedc1
    CD_big_C1 = np.row_stack((C1_D_0,C1_D_6,C1_D_12,C1_D_18))
    CD_big_C1 = CD_big_C1 + V_big_C1

    CD_big_P2 = np.row_stack((P2_D_0,P2_D_6,P2_D_12,P2_D_18))
    CD_big_P2 = CD_big_P2 + V_big_P2

    # Corrected reciver coordinates

 
    #Corrected Reciver coordinates Calculated With C1 (Big adjustment)
    kx_C1 = G_x + float(x_big_C1[0])
    ky_C1 = G_y + float(x_big_C1[1])
    kz_C1 = G_z + float(x_big_C1[2])

    #Corrected Reciver coordinates Calculated With C1 (Big adjustment)
    kx_P2 = G_x + float(x_big_P2[0])
    ky_P2 = G_y + float(x_big_P2[1])
    kz_P2 = G_z + float(x_big_P2[2])

    #Same values for c1 and p2
    N_big_inv = np.array(N_big_inv)
    Qx1x1_big = N_big_inv[0][0]
    Qy1y1_big = N_big_inv[1][1]
    Qz1z1_big = N_big_inv[2][2]

    #calculations for C1 code
    #V^TV
    vtv_C1 = float(np.transpose(V_big_C1)*(V_big_C1))

    #Level of freedom f = n - u da u value is 7. So shape of x matrix
    S0_big_C1 = (vtv_C1/(len(CD_big_C1)-len(x_big_C1)))**0.5 

    # Q values same for C1 Codes
    Sx_big_C1 = S0_big_C1*(Qx1x1_big)**0.5
    Sy_big_C1 = S0_big_C1*(Qy1y1_big)**0.5
    Sz_big_C1 = S0_big_C1*(Qz1z1_big)**0.5

    #calculations for P2 code
    #V^TV
    vtv_P2 = float(np.transpose(V_big_P2)*(V_big_P2))

    #Level of freedom f = n - u da u value is 7. So shape of x matrix
    S0_big_P2 = (vtv_P2/(len(CD_big_P2)-len(x_big_P2)))**0.5 

    # Q values same for P2 Codes
    Sx_big_P2 = S0_big_P2*(Qx1x1_big)**0.5
    Sy_big_P2 = S0_big_P2*(Qy1y1_big)**0.5
    Sz_big_P2 = S0_big_P2*(Qz1z1_big)**0.5

 
    psd_big_C1 = (Sx_big_C1**2+Sy_big_C1**2+Sz_big_C1**2)**0.5
    psd_big_P2 = (Sx_big_P2**2+Sy_big_P2**2+Sz_big_P2**2)**0.5

    #same for c1 and p2
    PDOP_big = (Qx1x1_big+Qy1y1_big+Qz1z1_big)**2


    uydu0_tn = np.array(uydu0_t)
    uydu6_tn = np.array(uydu6_t)
    uydu12_tn = np.array(uydu12_t)
    uydu18_tn = np.array(uydu18_t)

    #uydulara ait x,y,z,d
    uydu0_tn.shape = (4,len(uydu0_x))
    uydu0_tn = np.transpose(uydu0_tn) #son hane ışık hızı ile çarpılmış
    uydu6_tn.shape = (4,len(uydu6_x))
    uydu6_tn = np.transpose(uydu6_tn)
    uydu12_tn.shape = (4,len(uydu12_x))
    uydu12_tn = np.transpose(uydu12_tn)
    uydu18_tn.shape = (4,len(uydu18_x))
    uydu18_tn = np.transpose(uydu18_tn)

 
    #adcusted C1
    CDD_big_C1 = CD_big_C1+0

    #same for c1 and p2 codes
    uyduxyz_all = np.row_stack((uydu0_tn,uydu6_tn,uydu12_tn,uydu18_tn))

    #Calculation for C1
    for i in range(len(uyduxyz_all)):
        if i < len(C1_D_0): 
            CDD_big_C1[i] = ((uyduxyz_all[i][0] - kx_C1)**2 + (uyduxyz_all[i][1] - ky_C1)**2 + (uyduxyz_all[i][2] - kz_C1)**2)**0.5 + float(x_big_C1[3])*299792458
            
        elif i < len(C1_D_0)+len(C1_D_6):
            CDD_big_C1[i] = ((uyduxyz_all[i][0] - kx_C1)**2 + (uyduxyz_all[i][1] - ky_C1)**2 + (uyduxyz_all[i][2] - kz_C1)**2)**0.5 + float(x_big_C1[4])*299792458

        elif i < len(C1_D_0)+len(C1_D_6)+len(C1_D_12):
            CDD_big_C1[i] = ((uyduxyz_all[i][0] - kx_C1)**2 + (uyduxyz_all[i][1] - ky_C1)**2 + (uyduxyz_all[i][2] - kz_C1)**2)**0.5 + float(x_big_C1[5])*299792458

        elif i < len(C1_D_0)+len(C1_D_6)+len(C1_D_12)+len(C1_D_18):
            CDD_big_C1[i] = ((uyduxyz_all[i][0] - kx_C1)**2 + (uyduxyz_all[i][1] - ky_C1)**2 + (uyduxyz_all[i][2] - kz_C1)**2)**0.5 + float(x_big_C1[6])*299792458
        

    #adcusted P2
    CDD_big_P2 = CD_big_P2+0

    #same for P2 and p2 codes
    uyduxyz_all = np.row_stack((uydu0_tn,uydu6_tn,uydu12_tn,uydu18_tn))

    #Calculation for P2
    for i in range(len(uyduxyz_all)):
        if i < len(P2_D_0): 
            CDD_big_P2[i] = ((uyduxyz_all[i][0] - kx_P2)**2 + (uyduxyz_all[i][1] - ky_P2)**2 + (uyduxyz_all[i][2] - kz_P2)**2)**0.5 + float(x_big_P2[3])*299792458
            
        elif i < len(P2_D_0)+len(P2_D_6):
            CDD_big_P2[i] = ((uyduxyz_all[i][0] - kx_P2)**2 + (uyduxyz_all[i][1] - ky_P2)**2 + (uyduxyz_all[i][2] - kz_P2)**2)**0.5 + float(x_big_P2[4])*299792458

        elif i < len(P2_D_0)+len(P2_D_6)+len(P2_D_12):
            CDD_big_P2[i] = ((uyduxyz_all[i][0] - kx_P2)**2 + (uyduxyz_all[i][1] - ky_P2)**2 + (uyduxyz_all[i][2] - kz_P2)**2)**0.5 + float(x_big_P2[5])*299792458

        elif i < len(P2_D_0)+len(P2_D_6)+len(P2_D_12)+len(P2_D_18):
            CDD_big_P2[i] = ((uyduxyz_all[i][0] - kx_P2)**2 + (uyduxyz_all[i][1] - ky_P2)**2 + (uyduxyz_all[i][2] - kz_P2)**2)**0.5 + float(x_big_P2[6])*299792458
        

    # Diffrance values is so small so correction is succesful
    Diffrance_C1 = CDD_big_C1-CD_big_C1

    # Diffrance values is so small so correction is succesful
    Diffrance_P2 = CDD_big_P2-CD_big_P2

    return kx_C1, ky_C1, kz_C1, psd_big_C1, PDOP_big


#ista0070.20o
#igs20872.sp3
#main_gps_calc("ista0070.20o","igs20872.sp3")
