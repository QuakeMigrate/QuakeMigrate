#!/bin/bash

gcc -shared -fPIC -std=gnu99 onset.c cmscan.c levinson.c -fopenmp -O0 -o ../cmslib.so

