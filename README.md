# BCI-Project

##Objective
To analyze the EEG signals from the scalp near the visual cortex to infer what a person is seeing.

##People
gihanjayatilaka@eng.psn.ac.lk , harshana.w@eng.pdn.ac.lk, suren.sri@eng.pdn.ac.lk, anu321willamuna@gmail.com, nuwanjaliyagoda@eng.pdn.ac.lk, hiruna72@gmail.com


##Getting started

###Python and libraries
Python3, Numpy, Scipy, Keras, Matplotlib


###OpenBCI
download processing

test
  https://github.com/OpenBCI/OpenBCI_Processing

if not connecting.
test
  https://github.com/OpenBCI/OpenBCI_Radio_Config_Utility
  press autoconnect followed by autoscan.
  should connect now.

now python !

test
  https://github.com/OpenBCI/OpenBCI_LSL
  close other processing streams before runnig "openbci_lsl.py --stream".
  start - /start.
  stop - /stop.
  exit - /exit.
  
test - (real stuff)
  https://github.com/NeuroTechX/bci-workshop/blob/master/INSTRUCTIONS.md
  this repo has feature vectors, SVM and a bit of ML, but lots of bugs.
  
  install required packages. no need of conda.
  look out for python 2. tkinter should be install using sudo apt-get)
  
test
  run "openbci_lsl.py --stream" as before and '/start' the stream.
  run "exercise_01.py" (should get errors in the first run. get over it!)
  
  
#tried and worked in xubuntu.
