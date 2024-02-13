#!/bin/bash
# Execute this file to recompile locally
c++ -Wall -shared -fPIC -std=c++11 -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only -I/Users/nicole/anaconda3/envs/FEniCS-OpInf/include -I/Users/nicole/anaconda3/envs/FEniCS-OpInf/include/eigen3 -I/Users/nicole/anaconda3/envs/FEniCS-OpInf/.cache/dijitso/include dolfin_expression_8f6d3c13208c5aeb2af1dc7b0ec1341d.cpp -L/Users/nicole/anaconda3/envs/FEniCS-OpInf/lib -L/Users/nicole/anaconda3/envs/FEniCS-OpInf/Users/nicole/anaconda3/envs/FEniCS-OpInf/lib -L/Users/nicole/anaconda3/envs/FEniCS-OpInf/.cache/dijitso/lib -Wl,-rpath,/Users/nicole/anaconda3/envs/FEniCS-OpInf/.cache/dijitso/lib -lpmpi -lmpi -lmpicxx -lpetsc -lslepc -lhdf5 -lboost_timer -ldolfin -Wl,-install_name,/Users/nicole/anaconda3/envs/FEniCS-OpInf/.cache/dijitso/lib/libdijitso-dolfin_expression_8f6d3c13208c5aeb2af1dc7b0ec1341d.so -olibdijitso-dolfin_expression_8f6d3c13208c5aeb2af1dc7b0ec1341d.so