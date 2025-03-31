// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_EIGENSOLVER_LAPACK_DECL_HPP
#define _FROSCH_EIGENSOLVER_LAPACK_DECL_HPP

//Teuchos
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"

//#include <ShyLU_DDFROSch_config.h>

// FROSch
#include "FROSch_EigenSolverFactory_fwd_decl.hpp" // forward declaration
#include "FROSch_EigenSolver_decl.hpp"

namespace FROSch {

/**
* @brief LAPACK via Teuchos: Direct eigen solver for generalized eigenvalue problems.
*/

template <typename OpA, 
          typename OpB, 
          typename SC = Tpetra::Details::DefaultTypes::scalar_type, 
          typename LO = Tpetra::Details::DefaultTypes::local_ordinal_type,
          typename GO = Tpetra::Details::DefaultTypes::global_ordinal_type, 
          typename NO = Tpetra::Details::DefaultTypes::node_type>
class EigenSolver_LAPACK : public EigenSolver<OpA, OpB, SC, LO, GO, NO> {

    friend class EigenSolverFactory<OpA, OpB, SC, LO, GO, NO>;

  private:
    using ParameterListPtr = Teuchos::RCP<Teuchos::ParameterList>;

    // Constructor
    EigenSolver_LAPACK(OpA opA,
                       OpB opB,
                       const ParameterListPtr parameterList,
                       Teuchos::RCP<std::vector<SC>> &eigenvalues_ptr,
                       Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &eigenvectors_ptr,
                       Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &choleskyFactor_ptr);

    /**
     * @brief Solve symmetric positive definite generalized eigenvalue problem with LAPACK.
     *        Input matrices WILL NOT be overwritten by LAPACK.
     *
     * @details
     * Solve A*X = B, where A and B are symmetric, and B is positive definite.
     * Use LAPACK via Teuchos for the solution.
     *
     * @param[in,out] matrixA_ptr   Will contain pointer to matrix of eigenvectors (in the columns).
     *                              The eigenvectors Z are normalized s.t. Z^T*B*Z = I = identity matrix.
     * @param[in,out] matrixB_ptr   Will contain pointer to matrix L of cholesky factorization B = L*L^T.
     * @param[in] computeOnlyEigenvalues   If true, matrixA will not contain eigenvectors.
     *                                     Computing only eigenvalues will not affect that matrixA and
     *                                     matrixB will be overwritten. See LAPACK documentation for details.
     *
     * @return Eigenvalues
     */
    Teuchos::RCP<std::vector<SC>>
    solveSymmPosDefGEVP_LAPACK_nondestructive(Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &matrixA_ptr,
                                              Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &matrixB_ptr,
                                              const bool computeOnlyEigenvalues = false);

    /**
     * @brief Solve symmetric positive definite generalized eigenvalue problem with LAPACK.
     *        Input matrices WILL be overwritten by LAPACK.
     *
     * @details
     * Solve A*X = B, where A and B are symmetric, and B is positive definite.
     * Use LAPACK via Teuchos for the solution.
     *
     * @param[in,out] matrixA   Will be overwritten by LAPACK with eigenvectors in the columns.
     *                          The eigenvectors Z are normalized s.t. Z^T*B*Z = I = identity matrix..
     * @param[in,out] matrixB   Will be overwritten by LAPACK with L of cholesky factorization B = L*L^T.
     * @param[in] computeOnlyEigenvalues   Computing only eigenvalues will not affect that
     *                                     matrixA and matrixB will be overwritten.
     *
     * @return Eigenvalues
     */
    Teuchos::RCP<std::vector<SC>>
    solveSymmPosDefGEVP_LAPACK_destructive(Teuchos::SerialDenseMatrix<LO, SC> &matrixA,
                                           Teuchos::SerialDenseMatrix<LO, SC> &matrixB,
                                           const bool computeOnlyEigenvalues = false);
};

} // namespace FROSch

#endif
