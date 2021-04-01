#!/usr/bin/env bash

set -e # fail on first error

if conda --version > /dev/null 2>&1; then
   echo "conda appears to alreaday be installed"
   exit 0
 fi

PYTHON_VERSION=${PYTHON_VERSION:-3.5} # if no python specified, use 3.5

if [ "$(uname)" == "Darwin" ]; then
  URL_OS="MacOSX"
elif [ "$(expr substr "$(uname -s)" 1 5)" == "Linux" ]; then
  URL_OS="Linux"
elif [ "$(expr substr "$(uname -s)" 1 10)" == "MINGW32_NT" ]; then
  URL_OS="Windows"
fi

echo "Downloading miniconda for $URL_OS"
DOWNLOAD_PATH="miniconda.sh"

if [ ${PYTHON_VERSION} == "2.7" ]; then
  wget http://repo.continuum.io/miniconda/Miniconda-latest-$URL_OS-x86_64.sh -O ${DOWNLOAD_PATH};
  INSTALL_FOLDER="$HOME/minconda2"
else
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-$URL_OS-x86_64.sh -O ${DOWNLOAD_PATH};
  INSTALL_FOLDER="$HOME/minconda3"
fi


echo "Installing miniconda for python-$PYTHON_VERSION to $INSTALL_FOLDER"
# install miniconda to home folder
bash ${DOWNLOAD_PATH} -b -p $INSTALL_FOLDER

# tidy up
rm ${DOWNLOAD_PATH}

export PATH="$INSTALL_FOLDER/bin:$PATH"
echo "Adding $INSTALL_FOLDER to PATH.  Consider adding it in your .rc file as well."
conda update -q -y conda
