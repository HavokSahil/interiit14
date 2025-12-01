#!/bin/bash

tarname="pilot-site-logs.tar.xz"
directory="pilot-site-logs"

if [[ -d "$directory" ]]; then
    echo -n 
else
    echo "directory pilot-site-logs does not exist"
    exit
fi

if [[ -e "$tarname" ]]; then
    echo "WARNING: the file $tarname already exists. Remove and continue (y/n)"
    read response
    case $response in
        y|Y)
            rm -rf "$tarname"
            ;;
        *)
            exit
    esac
fi

echo "Compressing $directory to $tarname..."
tar -cJf "$tarname" "$directory/"
echo "done"
