#ifndef _SVD_FRM_SVDLIB_H_
#define _SVD_FRM_SVDLIB_H_

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>
extern "C" {
  #include "svdlib.h"
}

#include "GKlib.h"

void svdFrmSvdlibCSR(gk_csr_t *mat, int rank, Eigen::MatrixXf& U,
                Eigen::MatrixXf& V, bool pureSVD);

#endif

