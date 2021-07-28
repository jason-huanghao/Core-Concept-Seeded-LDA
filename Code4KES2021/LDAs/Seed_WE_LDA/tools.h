#pragma once
#include <math.h>
#include <vector>
#include <iostream>
using namespace std;

void init_matrix(int **M, int R, int C, int value);
void init_vector(int *V, int C, int value);

void norm(double *V, int L, double sum);
void softmax(double *V, int L);


int index_up_traingle(int x, int y, int n);
int index_down_traingle(int x, int y);

void print_matrix(vector<vector<double>> & matrix);
//Eigen::MatrixXd ConvertVector2Matrix(vector<vector<double>> &data);
//Eigen::MatrixXd ConvertArray2Matrix(double ** data, int row, int col);


typedef unsigned long uint32;
uint32 randomMT(void);
void seedMT(uint32 seed);

#define unif() ((double) rand()) / ((double) RAND_MAX)
int mult_sample(double* val, double norm_sum);