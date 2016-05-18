#include "svdFrmsvdlib.h"


void svdFrmSvdlibCSR(gk_csr_t *mat, int rank, Eigen::MatrixXf& U, 
    Eigen::MatrixXf& V, bool pureSVD) {
  int nnz = 0;
  int u, i, j, item, jj;
  for (u = 0; u < mat->nrows; u++) {
    nnz += mat->rowptr[u+1] - mat->rowptr[u];
  }
  std::cout << "\nsvd mat nnz: " << nnz << std::endl;
  std::cout << "\nsvd mat nrows: " << mat->nrows << " ncols: " << mat->ncols 
    << std::endl; 
  std::unique_ptr<smat> ipMat(new smat());
  std::unique_ptr<long[]> pointr(new long[mat->ncols+1]);
  ipMat->pointr = pointr.get();
  std::unique_ptr<long[]> rowind(new long[nnz]);
  ipMat->rowind = rowind.get();
  std::unique_ptr<double[]> value(new double[nnz]);
  ipMat->value = value.get();
  
  ipMat->rows = mat->nrows;
  ipMat->cols = mat->ncols;
  ipMat->vals = nnz;

  for (item = 0; item < mat->ncols; item++) {
    pointr[item] = mat->colptr[item];
    for (jj = mat->colptr[item]; jj < mat->colptr[item+1]; jj++) {
      rowind[jj] = mat->colind[jj];
      value[jj] = mat->colval[jj];
    }
  }
  pointr[item] = mat->colptr[item];

  //compute top-rank svd, returns pointer to svdrec
  SVDRec svd = svdLAS2A(ipMat.get(), rank); 
  std::cout << "\nDimensionality: " << svd->d;
  std::cout << "\nSingular values: ";
  for (i = 0; i < rank; i++) {
    std::cout << svd->S[i]  << " ";
  }
  
  std::cout << "\nUt nrows: " << svd->Ut->rows << " ncols: " << svd->Ut->cols;
  //copy singular vectors to uFac
  for (u = 0; u < mat->nrows; u++) {
    for (j = 0; j < rank; j++) {
      U(u, j) = svd->Ut->value[j][u];
    }
  }

  std::cout << "\nVt nrows: " << svd->Vt->rows << " ncols: " << svd->Vt->cols;
  //copy singular vectors to iFac
  for (item = 0; item < mat->ncols; item++) {
    for (j = 0; j < rank; j++) {
      if (pureSVD) {
        V(item, j) = svd->Vt->value[j][item]*svd->S[j];
      } else {
        V(item, j) = svd->Vt->value[j][item];
      }
    }
  }
  
  //free svdrec
  svdFreeSVDRec(svd);
}


