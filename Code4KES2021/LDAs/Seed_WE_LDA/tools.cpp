#include "tools.h"

int mult_sample(double* vals, double norm_sum)
{
	double rand_sample = unif() * norm_sum;
	double tmp_sum = 0;
	int j = 0;
	while (tmp_sum < rand_sample || j == 0) {
		tmp_sum += vals[j];
		j++;
	}
	return j - 1;
}

void init_matrix(int **M, int R, int C, int value) {
	for (int r = 0; r < R; r++)
		for (int c = 0; c < C; c++)
			M[r][c] = value;
}

void init_vector(int *V, int C, int value) {
	for (int c = 0; c < C; c++) V[c] = value;
}

void norm(double *V, int L, double sum) {
	for (int k = 0; k < L; k++) sum += V[k];
	for (int k = 0; k < L; k++) V[k] /= sum;
}

void softmax(double *V, int L) {
	double sum = 0.0;
	for (int k = 0; k < L; k++) {
		V[k] = exp(V[k]);
		sum += V[k];
	}
	for (int k = 0; k < L; k++)
		V[k] /= sum;
}

int index_up_traingle(int x, int y, int n) {
	// for a up traingle matrix without diagonal element, x < y < n
	if (x == y) return -1;

	if (y < x) {
		int t = x; 
		x = y;
		y = t;
	}

	return x * (2 * n - x + 1) / 2 + y - 2 * x - 1;
}

int index_down_traingle(int x, int y) {
	// for a down traingle matrix without diagonal element n > x > y 
	if (x == y) return -1;

	if (y > x) {
		int t = x;
		x = y;
		y = t;
	}

	return x * (x + 1) / 2 + y - x;
}

void print_matrix(vector<vector<double>> & matrix) {
	vector<double> sums;
	for (int i = 0; i < matrix.size(); i++) {
		double sum = 0.0;
		for (int j = 0; j < matrix[i].size(); j++) {
			cout << matrix[i][j] << " ";
			sum += matrix[i][j];
		}
		cout << endl;
		sums.push_back(sum);
	}
	for (int v = 0; v < sums.size(); v++) {
		cout << sums[v] << endl;
	}
}

//Eigen::MatrixXd ConvertVector2Matrix(vector<vector<double>> &data) {
//	int row = data.size(), col = data[0].size();
//	Eigen::MatrixXd eMatrix(row, col);
//	for (int i = 0; i < row; i++) {
//		eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], col);
//	}
//	return eMatrix;
//}
//
//Eigen::MatrixXd ConvertArray2Matrix(double ** data, int row, int col) {
//	Eigen::MatrixXd eMatrix(row, col);
//	for (int i = 0; i < row; i++) {
//		eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], col);
//	}
//	return eMatrix;
//}

#ifndef COKUS_H_
#define COKUS_H_
#define NN              (624)                 // length of state vector
#define MM              (397)                 // a period parameter
#define KK              (0x9908B0DFU)         // a magic constant
#define hiBit(u)       ((u) & 0x80000000U)   // mask all but highest   bit of u
#define loBit(u)       ((u) & 0x00000001U)   // mask all but lowest    bit of u
#define loBits(u)      ((u) & 0x7FFFFFFFU)   // mask     the highest   bit of u
#define mixBits(u, v)  (hiBit(u)|loBits(v))  // move hi bit of u to hi bit of v

static uint32   state[NN + 1];     // state vector + 1 extra to not violate ANSI C
static uint32   *next_;          // next_ random value is computed from here
static int      left_ = -1;      // can *next_++ this many times before reloading

#endif


void seedMT(uint32 seed)
{
	register uint32 x = (seed | 1U) & 0xFFFFFFFFU, *s = state;
	register int    j;

	for (left_ = 0, *s++ = x, j = NN; --j;
		*s++ = (x *= 69069U) & 0xFFFFFFFFU);
}


uint32 reloadMT(void)
{
	register uint32 *p0 = state, *p2 = state + 2, *pM = state + MM, s0, s1;
	register int    j;

	if (left_ < -1)
		seedMT(4357U);

	left_ = NN - 1, next_ = state + 1;

	for (s0 = state[0], s1 = state[1], j = NN - MM + 1; --j; s0 = s1, s1 = *p2++)
		*p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? KK : 0U);

	for (pM = state, j = MM; --j; s0 = s1, s1 = *p2++)
		*p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? KK : 0U);

	s1 = state[0], *p0 = *pM ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? KK : 0U);
	s1 ^= (s1 >> 11);
	s1 ^= (s1 << 7) & 0x9D2C5680U;
	s1 ^= (s1 << 15) & 0xEFC60000U;
	return(s1 ^ (s1 >> 18));
}


uint32 randomMT(void)
{
	uint32 y;

	if (--left_ < 0)
		return(reloadMT());

	y = *next_++;
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9D2C5680U;
	y ^= (y << 15) & 0xEFC60000U;
	y ^= (y >> 18);
	return(y);
}