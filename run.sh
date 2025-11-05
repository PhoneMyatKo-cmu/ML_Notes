#!/bin/bash

set -xe

clang -Wall -Wextra -o main two_params.c -lm
./main
