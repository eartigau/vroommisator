from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

file1 = 'composite_transmission_N14_step10_old.csv'
file2 = 'extracted_transmission_data_updated_coatings.csv'

front_end_transmission = 'vroomm-frontend-transmission.csv'

tbl_emccd = Table.read('emcc_interpolated.csv')
tbl_emccd['transmission']/=100.0

tbl_front = Table.read(front_end_transmission)
tbl_front['wavelength']*=1000

n = 14  # number of surfaces

tbl1_tmp = Table.read(file1)
tbl1_tmp['loss'] = tbl1_tmp['transmission_loss']
tbl2_tmp = Table.read(file2)
tbl2_tmp['loss'] = tbl2_tmp['transmission']
tbl2_tmp['loss']/=100.0

tbl1_tmp['transmission'] = (1 - tbl1_tmp['loss'])**n
tbl2_tmp['transmission'] = (1 - tbl2_tmp['loss'])**n

wave = np.arange(360, 931, 1)
tbl1 = Table()
tbl1['wavelength'] = wave
spline1 = UnivariateSpline(tbl1_tmp['wavelength'], tbl1_tmp['transmission'], s=0)#(wave)
tbl1['transmission'] = spline1(wave)
tbl2 = Table()
tbl2['wavelength'] = wave
spline2 = UnivariateSpline(tbl2_tmp['wavelength'], tbl2_tmp['transmission'], s=0)#(wave)
tbl2['transmission'] = spline2(wave)

tbl3 = Table()
tbl3['wavelength'] = wave
spline3 = UnivariateSpline(tbl_front['wavelength'], tbl_front['transmission'], s=0)#(wave)
tbl3['transmission'] = spline3(wave)
                           

tbl4 = Table()
tbl4['wavelength'] = wave
spline4 = UnivariateSpline(tbl_emccd['wavelength'], tbl_emccd['transmission'], s=0)(wave)
tbl4['transmission'] = spline4

plt.plot(tbl1['wavelength'], tbl1['transmission'],color='red', linestyle = '--', label = 'Back-end #1')
plt.plot(tbl2['wavelength'], tbl2['transmission'],color = 'blue', linestyle = '--', label = 'Back-end #2')

plt.plot(tbl2['wavelength'], tbl3['transmission'],  color='green', label = 'Front-end')
plt.plot(tbl4['wavelength'], tbl4['transmission'],  color='orange', label = 'EMCCD')

plt.plot(tbl1['wavelength'], tbl1['transmission']*tbl3['transmission']*tbl4['transmission'],  linestyle='-',color='red',label = 'Back-end #1 + front-end + EMCCD')

plt.plot(tbl2['wavelength'], tbl2['transmission']*tbl3['transmission']*tbl4['transmission'],   linestyle='-',color='blue', label = 'Back-end #2 + front-end + EMCCD')

plt.xlabel('Wavelength')
plt.ylabel('Transmission (%)')
plt.grid(True)
plt.legend()
plt.xlim([360,930])
#plt.ylim([0,0.025])
plt.savefig('compare_transmission_spectra.png', dpi=150)
plt.show()

tbl = Table()
tbl['wavelength'] = tbl1['wavelength']
tbl['transmission'] = tbl1['transmission']*tbl3['transmission']*tbl4['transmission']

tbl.write('combined_transmission_spectrum.csv', format='csv', overwrite=True)