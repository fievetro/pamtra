#!/bin/bash

# This is MPIM version of the install script for pamtra on Levante.

#################################################################################
# First, load the required modules.
echo "Purging modules and loading required ones."

module purge
module load git
spack load /bcn7mbu # gcc
spack load /l2ulgpu # netcdf
spack load /fnfhvr6 # fftw
spack load /tpmfvw  # openblas

echo "Loading python"
which python3

#################################################################################

# Create a PAMTRA_DATADIR, change the PAMTRA_DATA_FOLDER variable if you want it to point somewhere else
THISDIR=$(pwd)
echo $THISDIR
PAMTRA_DATA_FOLDER=$THISDIR/pamtra_data/
mkdir -p $PAMTRA_DATA_FOLDER

# If you do not need pamtra_data on scattering and emissivity, or the example data, you can comment out the relevant lines from this block
echo "Downloading data"
wget -q -O data.tar.bz2 https://uni-koeln.sciebo.de/s/As5fqDdPCOx4JbS/download
wget -q -O example_data.tar.bz2 https://uni-koeln.sciebo.de/s/28700CuFssmin8q/download
tar -xjvf example_data.tar.bz2 -C $PAMTRA_DATA_FOLDER
tar -xjvf data.tar.bz2 -C $PAMTRA_DATA_FOLDER
rm example_data.tar.bz2
rm data.tar.bz2

# This is needed, clean bashrc and bash_profile
sed '/### >>> pamtra >>> ###/,/### <<< pamtra <<< ###/d' ${HOME}/.bashrc > cleanbash
mv cleanbash ${HOME}/.bashrc

# Edit bashrc and bash profile
echo "### >>> pamtra >>> ###" >> ${HOME}/.bashrc
echo "export PYTHONPATH=$PYTHONPATH:$HOME/lib/python" >> ${HOME}/.bashrc
echo "export PAMTRA_DATADIR=${PAMTRA_DATA_FOLDER}" >> ${HOME}/.bashrc
echo "export OPENBLAS_NUM_THREADS=1" >> ${HOME}/.bashrc # this is important only for the openblas option
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/spack-levante/openblas-0.3.18-tpmfvw/lib/:/sw/spack-levante/fftw-3.3.10-fnfhvr/lib/" >> ${HOME}/.bashrc
echo "### <<< pamtra <<< ###" >> ${HOME}/.bashrc

# Clean the folder of compiled files
make clean

# Now, let's modify the Makefile for openblas
sed 's/-llapack//g' Makefile > Makefile.levante
sed -i 's%-lblas% -L/sw/spack-levante/openblas-0.3.18-tpmfvw/lib/ -L/sw/spack-levante/fftw-3.3.10-fnfhvr/lib/ -lopenblas%g' Makefile.levante

# and install
make -f Makefile.levante
make pyinstall -f Makefile.levante

# Make pamtra immediately available, even on the login node
source $HOME/.bashrc

# Write to a readme
echo "If no error message was printed, you should be good to go! Thank you for using pamtra"
echo "Remember to load the python module if you want to use pamtra in a batch script!"
