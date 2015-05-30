
float dotProd(float *u, float *v, int sz) {
  float prod = 0;
  int i;
  for (i = 0; i < sz; i++) {
    prod += u[i]*v[i];
  }
  return prod;
}
