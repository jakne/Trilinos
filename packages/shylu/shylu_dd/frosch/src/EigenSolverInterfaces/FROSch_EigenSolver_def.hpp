
#ifndef _FROSCH_EIGENSOLVER_DEF_HPP
#define _FROSCH_EIGENSOLVER_DEF_HPP

#include "FROSch_EigenSolver_decl.hpp"
#include "FROSch_EigenSolverOperator_def.hpp"


namespace FROSch {

template< typename SC, typename LO, typename GO, typename NO, typename OpA, typename OpB >
EigenSolver< SC, LO, GO, NO, OpA, OpB >::EigenSolver(
                 OpA opA, 
						     OpB opB, 
						     ParameterListPtr parameterList,
                 Teuchos::RCP< std::vector<SC> > &eigenvalues_ptr,
                 Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> >& eigenvectors_ptr) : 
opA_ (Teuchos::rcp(new EigenSolverOperator<SC,LO,GO,NO>(opA))),
opB_ (Teuchos::rcp(new EigenSolverOperator<SC,LO,GO,NO>(opB))),
parameterList_ (parameterList)
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
