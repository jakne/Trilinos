// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_EIGENSOLVER_LAPACK_DEF_HPP
#define _FROSCH_EIGENSOLVER_LAPACK_DEF_HPP

#include "FROSch_EigenSolver_LAPACK_decl.hpp"

// Teuchos is a required dependency of FROSch.
// LAPACK is contained in Teuchos and, thus, always available.
// --> Our fallback eigensolver will always be LAPACK, specifically the
//     direct solver for generalized eigenvalue problems A*x = B,
//     where A and B are symmetric, and B is positive definite.
//     LAPACK: dsygv and ssygv
//     Documentation for double precision:
//        https://www.netlib.org/lapack/explore-html/d1/d39/group__hegv_gadc3e6dc69532c1233818df364b7de912.html#gadc3e6dc69532c1233818df364b7de912
#include "Teuchos_LAPACK.hpp"

namespace FROSch {

/*
  template< typename SC, typename LO, typename GO, typename NO >
  int DirectEigenSolver_LAPACK<SC,LO,GO,NO>::initialize()
  {
      this->IsInitialized_ = true;
      this->IsComputed_ = false;
      return 0;
  }

  template< typename SC, typename LO, typename GO, typename NO >
  int DirectEigenSolver_LAPACK<SC,LO,GO,NO>::compute()
  {
      FROSCH_ASSERT(this->IsInitialized_,"FROSch::DirectEigenSolver_LAPACK::compute()  Instance is not initialized.");
      // $$$ call LAPACK
      this->IsComputed_ = true;
      return 0;
  }
*/

template <typename OpA, typename OpB, typename SC, typename LO, typename GO, typename NO>
EigenSolver_LAPACK<OpA, OpB, SC, LO, GO, NO>::EigenSolver_LAPACK(
    OpA opA,
    OpB opB,
    const ParameterListPtr parameterList,
    Teuchos::RCP<std::vector<SC>> &eigenvalues_ptr,
    Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &eigenvectors_ptr,
    Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &choleskyFactor_ptr)
    : EigenSolver<OpA, OpB, SC, LO, GO, NO>(opA, opB, parameterList, eigenvalues_ptr, eigenvectors_ptr)
{
    FROSCH_ASSERT(!this->opA_.is_null(), "FROSch::EigenSolver_LAPACK: Operator A is null.");
    FROSCH_ASSERT(!this->opB_.is_null(), "FROSch::EigenSolver_LAPACK: Operator B is null.");
    FROSCH_ASSERT(this->opA_->isSerialDenseMatrix(),
                  "FROSch::EigenSolver_LAPACK: Operator A is not a serial dense matrix.");
    FROSCH_ASSERT(this->opB_->isSerialDenseMatrix(),
                  "FROSch::EigenSolver_LAPACK: Operator B is not a serial dense matrix.");

    Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> A = this->opA_->getSerialDenseMatrixPtr();
    Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> B = this->opB_->getSerialDenseMatrixPtr();

    eigenvalues_ptr = solveSymmPosDefGEVP_LAPACK_nondestructive(A, B);
    eigenvectors_ptr = A;
    choleskyFactor_ptr = B;
}

template <typename OpA, typename OpB, typename SC, typename LO, typename GO, typename NO>
Teuchos::RCP<std::vector<SC>> EigenSolver_LAPACK<OpA, OpB, SC, LO, GO, NO>::solveSymmPosDefGEVP_LAPACK_nondestructive(
    Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &matrixA_ptr,
    Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &matrixB_ptr,
    const bool computeOnlyEigenvalues)
{
    // LAPACK (SYGV) overwrites the passed matrices. --> Create copies of A and B.
    Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> matrixA_cpy_ptr =
        Teuchos::rcp(new Teuchos::SerialDenseMatrix<LO, SC>(*matrixA_ptr));
    Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> matrixB_cpy_ptr =
        Teuchos::rcp(new Teuchos::SerialDenseMatrix<LO, SC>(*matrixB_ptr));

    Teuchos::RCP<std::vector<SC>> eigval_ptr =
        solveSymmPosDefGEVP_LAPACK_destructive(*matrixA_cpy_ptr, *matrixB_cpy_ptr, false);

    matrixA_ptr = matrixA_cpy_ptr;
    matrixB_ptr = matrixB_cpy_ptr;

    return eigval_ptr;
}

//  template <typename SC = double, typename LO = int> $$$
template <typename OpA, typename OpB, typename SC, typename LO, typename GO, typename NO>
Teuchos::RCP<std::vector<SC>> EigenSolver_LAPACK<OpA, OpB, SC, LO, GO, NO>::solveSymmPosDefGEVP_LAPACK_destructive(
    Teuchos::SerialDenseMatrix<LO, SC> &matrixA,
    Teuchos::SerialDenseMatrix<LO, SC> &matrixB,
    const bool computeOnlyEigenvalues)
{
    // LAPACK: dsygv
    // https://www.netlib.org/lapack/explore-html/d1/d39/group__hegv_gadc3e6dc69532c1233818df364b7de912.html#gadc3e6dc69532c1233818df364b7de912
    Teuchos::LAPACK<int, SC> lapack;

    // Solve generalized eigenvalue problem using lapack's SYGV. If SC == double, then 
    // DSYGV( itype, jobz, uplo, n, A, lda, B, ldb, eigval, work, lwork, info ) is used.
    int itype = 1; // Solve A*x = lambda*B*x.
    char jobz;
    if (computeOnlyEigenvalues) {
        jobz = 'N'; // Compute eigenvalues.
    } else {
        jobz = 'V'; // Compute eigenvalues and eigenvectors.
    }
    char uplo = 'L';            // A,B symmetric --> Reference only their lower triangular data.
    int n = matrixA.numRows();  // size of matrix
    SC* A = matrixA.values();   // pointer to copy of A
    SC* B = matrixB.values();   // pointer to copy of B
    int ldA = matrixA.stride(); // leading dimension / stride of A
    int ldB = matrixB.stride(); // leading dimension / stride of B
    Teuchos::RCP<std::vector<SC>> eigval_ptr = Teuchos::rcp(new std::vector<SC>(matrixA.numRows())); // eigenvalues
    std::vector<SC> work(1); // Used to query LAPACK for optimal workspace size.
    int lwork = -1;          // Query LAPACK for optimal workspace size.
    int info;                // On OUT: ==0 then success;
                             //          <0 then error in arguments;
                             //          >0 then algorithm failed to converge.
                             // See LAPACK *SYGV documentation for details.

    // 1) Query LAPACK  for optimal workspace size.
    lapack.SYGV(itype, jobz, uplo, n, A, ldA, B, ldB, &(*eigval_ptr)[0], &work[0], lwork, &info);
    TEUCHOS_TEST_FOR_EXCEPTION(
        info != 0,
        std::runtime_error,
        "FROSch::EigenSolver::solveSymmPosDefGEVP_LAPACK_destructive(matrixA,matrixB,computeOnlyEigenvalues)"
        ": LAPACK SYGV failed to query optimal work size.");

    // The optimal workspace size is stored in the first element of 'work'.
    lwork = (int)work[0];
    work.resize(lwork);

    // 2) Solve generalized eigenvalue problem with LAPACK (dsygv). Eigenvalues in ascending order if successfull.
    lapack.SYGV(itype, jobz, uplo, n, A, ldA, B, ldB, &(*eigval_ptr)[0], &work[0], lwork, &info);

    // Analyse status value returned by LAPACK.
    if (info < 0) {
        // Illegal argument detected in the call to LAPACK.
        std::stringstream errMsg;
        errMsg << "Error. "
                  "FROSch::EigenSolver::solveSymmPosDefGEVP_LAPACK_destructive(matrixA,matrixB,computeOnlyEigenvalues)"
                  ": Illegal argument at position "
               << std::to_string(-info) << "while calling LAPACK.";
        FROSCH_ASSERT(false, errMsg.str());
    } else if (info > 0) {
        if (info <= n) {
            // LAPACK algorithm failed to converge.
            std::stringstream errMsg;
            errMsg
                << "Error. "
                   "FROSch::EigenSolver::solveSymmPosDefGEVP_LAPACK_destructive(matrixA,matrixB,computeOnlyEigenvalues)"
                   ": LAPACK did not converge. "
                   "LAPACK error code: "
                << std::to_string(info) << ".";
            FROSCH_ASSERT(false, errMsg.str());
        } else {
            // LAPACK detected that matrixB is not positive definite.
            std::stringstream errMsg;
            errMsg
                << "Error. "
                   "FROSch::EigenSolver::solveSymmPosDefGEVP_LAPACK_destructive(matrixA,matrixB,computeOnlyEigenvalues)"
                   ": matrixB (right-hand side of A*x = lambda*B*x) is not positive definite. "
                   "LAPACK return code: "
                << std::to_string(info)
                << ". "
                   "Matrix size: "
                << std::to_string(n) << ".";
            FROSCH_ASSERT(false, errMsg.str());
        }
    }

    return eigval_ptr;
}

} // namespace FROSch

#endif
