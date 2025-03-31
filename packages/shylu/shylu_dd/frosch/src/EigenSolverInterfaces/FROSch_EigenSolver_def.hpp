// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_EIGENSOLVER_DEF_HPP
#define _FROSCH_EIGENSOLVER_DEF_HPP

#include "FROSch_EigenSolverOperator_def.hpp"
#include "FROSch_EigenSolver_decl.hpp"

namespace FROSch {

template <typename OpA, typename OpB, typename SC, typename LO, typename GO, typename NO>
EigenSolver<OpA, OpB, SC, LO, GO, NO>::EigenSolver(OpA opA,
                                                   OpB opB,
                                                   ParameterListPtr parameterList,
                                                   Teuchos::RCP<std::vector<SC>> &eigenvalues_ptr,
                                                   Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &eigenvectors_ptr)
    : opA_(Teuchos::rcp(new EigenSolverOperator<SC, LO, GO, NO>(opA))),
      opB_(Teuchos::rcp(new EigenSolverOperator<SC, LO, GO, NO>(opB))), 
      parameterList_(parameterList)
{
    // $$$
    // @param[in] symmetrizeMatrices   Compute M = 0.5*(M + M^T) for matrixA and matrixB.
    // @param[in] symmetryOfMatrices_check   Check whether matrixA and matrixB are symmetric.
    // @param[in] symmetryOfMatrices_rel_tol   Relative tolerance used to check for symmetry of matrixA and matrixB. @See symmetryCheck().
    //const bool   symmetrizeMatrices = true;
    //const bool   symmetryOfMatrices_check = true;
    //const double symmetryOfMatrices_rel_tol = 1.0e-9;
}

/*
template<typename SC,class LO,class GO,class NO>
bool EigenSolver<SC,LO,GO,NO>::isInitialized() const
{
    return isInitialized_;
}

template<class SC,class LO,class GO,class NO>
bool EigenSolver<SC,LO,GO,NO>::isComputed() const
{
    return isComputed_;
}
*/

} // end namespace FROSch

#endif
