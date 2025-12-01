#!/bin/bash

tarname="pilot-site-logs.tar.xz"
directory="pilot-site-logs"

if [[ -e "$tarname" ]]; then
    echo -n 
else
    echo "archieve $tarname does not exist"
    exit
fi

if [[ -d "$directory" ]]; then
    echo "WARNING: the directory $directory already exists. Remove and continue (y/n)"
    read response
    case $response in
        y|Y)
            rm -rf "$directory"
            ;;
        *)
            exit
    esac
fi

echo "Decompressing the $tarname to $dirname ..."
tar -cvf "$tarname"
echo "done"

