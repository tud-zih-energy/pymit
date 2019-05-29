#!/bin/bash

rm -rf Dataset.pdf
rm -rf MADELON/
rm -rf madelon_valid.labels

if [ ! -e MADELON.zip ]
then
    wget http://clopinet.com/isabelle/Projects/NIPS2003/MADELON.zip
fi
unzip MADELON.zip MADELON/*

echo "python hjmi.py"
python hjmi.py
echo "python jmi.py"
python jmi.py
