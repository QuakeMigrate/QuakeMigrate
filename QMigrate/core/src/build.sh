#!/bin/bash

gcc -shared -fPIC -std=gnu99 QMigrate.c -fopenmp -O0 -o QMigrate.so

