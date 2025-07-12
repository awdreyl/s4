# small edit here

# This is a sample Python script.
#%%
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import S4 as S4
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
#%%
num=32
# n_aSi_940 = 3.72
# n_Si_940=3.48
n_SiO2_940 = 1.45359 
n_Si_940=3.59 
# n_SiO2_940 = 1.5359  
# n_aSi_940 = 3.68
# n_aSi_940 = 3.448
n_Si_1040 = 3.3955
# n_SiO2_940 = 1.45  
n_OV101=1.4
n_SiC_1000=2.5876
n_SiC_1550=2.5651
n_GaAs_1000=3.4734
n_GaAs_1550=3.3779
n_SiN_940=2.0164
# n_SiN_940=3.68
n_glass_940=1.5359
n_glass_1040=1.5058

t_OV101=0
t_box=3
t_PC=0.15
PC_str='%.0f'%(t_PC*1000)
h=t_PC
# a = 0.5
Type = 'A'
t_OV101_STR='%.0f'%(t_OV101*1000)                
def print_hi(name,a,ra):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name},a: {a},ra:{ra}')  # Press Ctrl+F8 to toggle the breakpoint.
start_value = 0.5
end_value = 0.51
step_size = 0.01

# Generate the list using numpy.arange()
lattice_value = np.arange(start_value, end_value, step_size)
#%%
if __name__ == '__main__':


    index_matrix=[]
    wvls = []
    Rs = []
    Ts = []
    As = []
    tE_real = []
    tE_imag = []
    rE_real = []
    rE_imag = []
    tE0_real = []
    tE0_imag = []
    rE0_real = []
    rE0_imag = []

    tE1_real = []
    tE1_imag = []
    rE1_real = []
    rE1_imag = []

    tE2_real = []
    tE2_imag = []
    rE2_real = []
    rE2_imag = []



    phaseT = []
    phaseT1 = []
    phaseT2=[]
    phaseR = []
    a=0.5
    h_glass=10
    tBuff=1
    tPath0=3

    # this changes the default date converter for better interactive plotting of dates:
    ra_list=[]
    Trans=[]
    Ref=[]
    ra_str=0.208
    ra_end=0.383
    step_size=0.001
    int_n=int((ra_end-ra_str)/step_size)
    index_str=1
    index_end=10
    index_step=((index_end-index_str)/int_n)

    a_str='%.0f'%(a*1000)
    i=0

    for ra in np.linspace(ra_str, ra_end, int_n):
        i=i+1
        print_hi('PyCharm',a,ra)
    # a=a_list
        S = S4.New(Lattice=((a, 0), (0, a)), NumBasis=num)
        S.AddMaterial(Name="Vacuum", Epsilon=1.0 + 0j)
        S.AddMaterial(Name="Si_1040", Epsilon=n_Si_1040 ** 2)
        S.AddMaterial(Name="Si_940", Epsilon=n_Si_940 ** 2)
        S.AddMaterial(Name="SiO2_940", Epsilon=n_SiO2_940**2)
        S.AddMaterial(Name="n_OV101_940", Epsilon=n_OV101**2)
        S.AddMaterial(Name="SiC_1550", Epsilon=n_SiC_1550**2)
        S.AddMaterial(Name="SiC_1000", Epsilon=n_SiC_1000**2)
        S.AddMaterial(Name="GaAs_1550", Epsilon=n_GaAs_1550**2)
        S.AddMaterial(Name="GaAs_1000", Epsilon=n_GaAs_1000**2)
        S.AddMaterial(Name="SiN_940", Epsilon=n_SiN_940**2)
        S.AddMaterial(Name="glass_940", Epsilon=n_glass_940**2)
        S.AddMaterial(Name="glass_1040", Epsilon=n_glass_1040**2)

        S.AddLayer(Name='topair', Thickness=0, Material='Vacuum')
        S.AddLayer(Name='AirBuff',Thickness=tBuff, Material='Vacuum')
        S.AddLayer(Name='AirPath',Thickness=tPath0, Material='Vacuum')
        S.AddLayer(Name='PC', Thickness=t_PC, Material='Si_1040')
        S.AddLayer(Name='Sub', Thickness=0, Material='glass_1040')
        
        # S0 for reference simulation
        S0 = S4.New(Lattice=((a, 0), (0, a)), NumBasis=num)

        S0.AddMaterial(Name="Vacuum", Epsilon=1.0 + 0j)

        S0.AddLayer(Name='topair', Thickness=0, Material='Vacuum')
        S0.AddLayer(Name='AirBuff',Thickness=tBuff, Material='Vacuum')
        S0.AddLayer(Name='AirPath',Thickness=tPath0, Material='Vacuum')
        S0.AddLayer(Name='PC', Thickness=t_PC, Material='Vacuum')
        S0.AddLayer(Name='Sub', Thickness=0, Material='Vacuum')

        r=ra*a

        wvl=1.04
        freq = 1 / wvl
        S.SetRegionCircle(
        Layer = 'PC',
        Material = 'Vacuum',
        Center = (0,0),
        # Radius = ra*a
        Radius = r
        )
        S.SetFrequency(freq)

        S.SetExcitationPlanewave(IncidenceAngles=(
            0, 0), sAmplitude=0 + 0j, pAmplitude=1 + 0j, Order=0)

        pyntfowtop, pyntbacktop = S.GetPoyntingFlux("topair", 0)
        pyntfowbot, pyntbackbot = S.GetPoyntingFlux("Sub", 0)

        Transmittance = np.abs(pyntfowbot) / np.abs(pyntfowtop)
        Reflectance = np.abs(pyntbacktop) / np.abs(pyntfowtop)
        Absorptance=1-Transmittance-Reflectance
        (rE, _) = S.GetFields(0, 0, tBuff)
        (tE, _) = S.GetFields(0, 0, tBuff + 2*tPath0+h)
        tE_real.append(tE[0].real)
        tE_imag.append(tE[0].imag)
        rE_real.append(rE[0].real)
        rE_imag.append(rE[0].imag)
        S0.SetFrequency(freq)
        S0.SetExcitationPlanewave(IncidenceAngles=(
        0, 0), sAmplitude=0 + 0j, pAmplitude=1 + 0j, Order=0)
        (rE0, _) = S0.GetFields(0, 0, tBuff)
        (tE0, _) = S0.GetFields(0, 0,tBuff + 2*tPath0+h)

        tE0_real.append(tE0[0].real)
        tE0_imag.append(tE0[0].imag)
        rE0_real.append(rE0[0].real)
        rE0_imag.append(rE0[0].imag)

        pht = np.angle(tE[0] /tE0[0] ) / np.pi
        # calcualting reflection phase
        phr = np.angle( (rE[0] - rE0[0])/tE0[0] ) / np.pi

###
        index_matrix.append(index_end-i*index_step)
        phaseT.append(pht)
        phaseR.append(phr)    
        print(ra)
        ra_list.append(ra)
        Trans.append(Transmittance)
        # Trans.append(Transmittance)
        Ref.append(Reflectance)
#%%
index_array=np.array(index_matrix)
ra_array=np.array(ra_list)
Trans_array=np.array(Trans)
Ref_array=np.array(Ref)
PhaseT_array=np.array(phaseT)
PhaseR_array=np.array(phaseR)
wl_str='%.0f'%(wvl*1000) 
    #%%
fig, ax1 = plt.subplots( )
line1=ax1.plot(ra_array,Trans_array,label="Trans") # trans
ax2 = ax1.twinx()
PhaseT_array_unwrap=np.unwrap(PhaseT_array, discont=2)
line3=ax2.plot(ra_array,PhaseT_array_unwrap-min(PhaseT_array_unwrap),'-',color='red',label="Trans Phase")
ax2.legend(bbox_to_anchor=(0., 0.15), loc=2, borderaxespad=0.,frameon=False)
ax1.legend(bbox_to_anchor=(0., 0.1), loc=2, borderaxespad=0.,frameon=False)

ax1.set_ylabel('Transmission and Reflection', fontsize=12.0)
ax2.set_ylabel('Phase (π)', fontsize=12.0)
ax1.set_xlabel('r/a', fontsize=12.0)
plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.ticker as ticker
# Assuming index_array and PhaseT_array_unwrap1 are already defined

# 1. Sort Data (Spline requires strictly increasing x-values)
sorted_indices = np.argsort(PhaseT_array_unwrap)
phase_sorted = PhaseT_array_unwrap[sorted_indices]
index_sorted = index_array[sorted_indices]

# 2. Create a Spline for Phase -> Index
spline_phase_to_index = UnivariateSpline(phase_sorted, index_sorted, s=0)  # Exact fit (s=0)

# 3. Generate Fitted Curve
phase_fit = np.linspace(min(phase_sorted), max(phase_sorted), 500)
index_fit = spline_phase_to_index(phase_fit)

# 4. Plot Original Data and Fitted Curve
plt.figure(figsize=(8, 6))
plt.plot(phase_sorted, index_sorted, 'o', color='red', label="Original Data (Phase vs. Index)")
plt.plot(phase_fit, index_fit, '-', color='blue', label="Spline Fit (Phase to Index)")
plt.xlabel('Phase (π)', fontsize=12.0)
plt.ylabel('PC slab index', fontsize=12.0)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
#%%
# 5. Extract Index from Input Phase
# Example: Use spline to find index for a specific phase value
input_phase = PhaseT_array_unwrap  # Example input phase
# PhaseT_array_unwrap=np.unwrap(PhaseT_array, period=2)

ff=np.pi*ra_array**2

# eff_indx_thoery=(2*ra_array*1+(1-2*ra_array)*n_aSi_940**2)**0.5
eff_indx_thoery=(ff*1**2+(1-ff)*n_Si_1040**2)**0.5
eff_phase=2*(eff_indx_thoery-1)*h/wvl

extracted_index = spline_phase_to_index(input_phase)
print(f"Extracted Index for Phase {input_phase}: {extracted_index}")
fig, ax1 = plt.subplots( )
line1=ax1.plot(ra_array,input_phase-min(input_phase),'-',label="Trans Phase with resonance")
line4=ax1.plot(ra_array,eff_phase+PhaseT_array_unwrap,label="Trans Phase without resonance")
ax2 = ax1.twinx()
line2=ax2.plot(ra_array,eff_indx_thoery,color='green',label="effective index without resonance")

line3=ax2.plot(ra_array,extracted_index,'-',color='red',label="effective index with resonance")
ax1.legend( loc=3, borderaxespad=0.,frameon=False,bbox_to_anchor=(0.43, 0.8))
ax2.legend( loc=3, borderaxespad=0.,frameon=False,bbox_to_anchor=(0.43, 0.9))

ax1.set_ylabel('Phase (π)', fontsize=12.0)
ax2.set_ylabel('effective index', fontsize=12.0)
ax1.set_xlabel('r/a', fontsize=12.0)
plt.tight_layout()
plt.show()
# Enable minor ticks
ax1.minorticks_on()
ax2.minorticks_on()

# Customize minor ticks' location
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))  # 4 minor ticks per interval

# Customize minor ticks' appearance
ax2.tick_params(axis='both', which='minor', length=4, color='r', direction='in', width=1)

#%%
# 5. Extract Index from Input Phase
# Example: Use spline to find index for a specific phase value
fig, ax1 = plt.subplots( )
line1=ax1.plot(ra_array,input_phase-min(input_phase),'-.',label="Trans Phase with resonance")
line4=ax1.plot(ra_array,eff_phase+PhaseT_array_unwrap,label="Trans Phase without resonance")
ax2 = ax1.twinx()
line2=ax2.plot(ra_array,eff_indx_thoery,color='green',label="effective index without resonance")

line3=ax2.plot(ra_array,extracted_index,'-',color='red',label="effective index with resonance")
ax1.legend( loc=3, borderaxespad=0.,frameon=False,bbox_to_anchor=(0.43, 0.8))
ax2.legend( loc=3, borderaxespad=0.,frameon=False,bbox_to_anchor=(0.43, 0.9))
ax1.set_ylabel('Phase (π)', fontsize=12.0)
ax2.set_ylabel('effective index', fontsize=12.0)
ax1.set_xlabel('r/a', fontsize=12.0)
plt.tight_layout()
plt.show()
# Enable minor ticks
ax1.minorticks_on()
ax2.minorticks_on()

# Customize minor ticks' location
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))  # 4 minor ticks per interval

# Customize minor ticks' appearance
ax2.tick_params(axis='both', which='minor', length=4, color='r', direction='in', width=1)

#%%
input_phase = PhaseT_array_unwrap  # Example input phase


ff=np.pi*ra_array**2

eff_indx_thoery=(ff*1**2+(1-ff)*n_Si_1040**2)**0.5
eff_phase=2*(eff_indx_thoery-1)*h/wvl
eff_L=wvl*(input_phase-min(input_phase))/(2*(eff_indx_thoery-1))

extracted_index = spline_phase_to_index(input_phase)
print(f"Extracted Index for Phase {input_phase}: {extracted_index}")
fig, ax1 = plt.subplots( )
line1=ax1.plot(ra_array,input_phase-min(input_phase),'-',label="Trans Phase with resonance")
ax2 = ax1.twinx()

line2=ax2.plot(ra_array,eff_L,color='green',label="effective thickness without resonance")

ra_array_flat = ra_array.flatten()  # Flatten ra_array to 1D
h_array = np.full_like(ra_array_flat, h)  # Create a 1D array filled with the constant h


line3 = ax2.plot(ra_array_flat, h_array, '-', color='red', label="Effective Index with Resonance")

ax1.legend( loc=3, borderaxespad=0.,frameon=False,bbox_to_anchor=(0., 0.05))
ax2.legend( loc=3, borderaxespad=0.,frameon=False,bbox_to_anchor=(0., 0.1))

ax1.set_ylabel('Phase (π)', fontsize=12.0)
ax2.set_ylabel('effective thickness (µm)', fontsize=12.0)
# plt.xlabel('Wavlength ($\mu$m)', fontsize=12.0)
ax1.set_xlabel('r/a', fontsize=12.0)
plt.tight_layout()
plt.show()
# Enable minor ticks
ax1.minorticks_on()
ax2.minorticks_on()

ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))  # 4 minor ticks per interval

# # Customize minor ticks' appearance
ax2.tick_params(axis='both', which='minor', length=4, color='r', direction='in', width=1)

#%%
DataSample=np.array([ra_list,input_phase,eff_indx_thoery,extracted_index])
SimpleDataFrame=pd.DataFrame(data=DataSample.T)
print(SimpleDataFrame)
SimpleDataFrame.to_csv('./Si'+PC_str+'nm_a'+a_str+'nm_wl'+wl_str+'nm_phase.txt',header = False,index = False)
SimpleDataFrame.to_csv('./sI'+PC_str+'nm_a'+a_str+'nm_wl'+wl_str+'nm_phase.csv',header = False,index = False)
