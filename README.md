# BCI-Project

## Objective
To analyze the EEG signals from the scalp near the visual cortex to infer what a person is seeing.

## Equipment
[Cyton + Daisy biosensing boards](https://shop.openbci.com/collections/openbci-products/products/cyton-daisy-biosensing-boards-16-channel).
This equipment (among other BCI, HCI equipment) can be borrowed from [Cambio wearable computing lab of Department of Computer Engineering, University of Peradeniya](https://cepdnaclk.github.io/sites/labs/wearable/).


## People
[Gihan](http://gihan.me) , [Harshana](http://teambitecode.com/people/harshana) , [Suren](http://teambitecode.com/people/suren) , [Anupamali](http://teambitecode.com/people/anupamali) , [Nuwan](http://teambitecode.com/people/nuwan), [Hiruna](https://github.com/hiruna72/)

## Getting started

### Python and libraries
Python3, Numpy, Scipy, Keras, Matplotlib


### OpenBCI
1. Download processing

2. test
  https://github.com/OpenBCI/OpenBCI_Processing

3. if not connecting.
test
  https://github.com/OpenBCI/OpenBCI_Radio_Config_Utility
  press autoconnect followed by autoscan.
  should connect now.

4. now python !

5. test
  https://github.com/OpenBCI/OpenBCI_LSL
  close other processing streams before runnig "openbci_lsl.py --stream".
  start - /start.
  stop - /stop.
  exit - /exit.
  
6. test - (real stuff)
  https://github.com/NeuroTechX/bci-workshop/blob/master/INSTRUCTIONS.md
  this repo has feature vectors, SVM and a bit of ML, but lots of bugs.
  
  install required packages. no need of conda.
  look out for python 2. tkinter should be install using sudo apt-get)
  
7. test
  run "openbci_lsl.py --stream" as before and '/start' the stream.
  run "exercise_01.py" (should get errors in the first run. get over it!)
  
  
**tried and worked in xubuntu.**

## Acknowledgements

The equipment were provided by [Cambio wearable computing lab](https://cepdnaclk.github.io/sites/labs/wearable/) at [Deaprtment of Computer Engineering](http://ce.pdn.ac.lk) , [Faculty of Engineering](http://eng.pdn.ac.lk) , [University of Peradeniya](http://www.pdn.ac.lk/)
