#include <Eigen/Dense>
#include <iostream>
#include <random>
using namespace std;

void genRand(mt19937& mt) {
  uniform_int_distribution<int> dist(101, 200);
  cout << dist(mt) << endl;
}

int main()
{
    Eigen::MatrixXf m(3,3);
      m << 1,2,3,
           4,5,6,
           7,8,9;
      cout << "Here is the matrix m:" << endl << m << endl;
      cout << "2nd Row: " << m.row(1) << endl;
      m.col(2) += 3 * m.col(0);
      cout << "After adding 3 times the first column into the third column, the matrix m is:\n";
      cout << m << endl;
      float p = m.row(0).dot(m.row(1));
      cout << "p = " << p << endl;

      //initialize random engine
      mt19937 mt(1);
      uniform_int_distribution<int> dist(0, 100);
      for (int i = 0; i < 50; i++) {
        genRand(mt);
      }
}

