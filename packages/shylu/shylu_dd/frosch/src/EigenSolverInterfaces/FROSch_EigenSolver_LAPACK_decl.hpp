#ifndef _FROSCH_EIGENSOLVER_LAPACK_DECL_HPP
#define _FROSCH_EIGENSOLVER_LAPACK_DECL_HPP

//Teuchos
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"

//#include <ShyLU_DDFROSch_config.h>

// FROSch
#include "FROSch_EigenSolver_decl.hpp"
#include "FROSch_EigenSolverFactory_fwd_decl.hpp"  // forward declaration


namespace FROSch {

  /**
   * @brief LAPACK via Teuchos: Direct eigen solver for generalized eigenvalue problems.
   */

  template < typename SC,// = double,
             typename LO,// = int,
             typename GO,// = DefaultGlobalOrdinal,
             typename NO,// = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType,
	           typename OpA,// $$$ default type?
	           typename OpB >
  class EigenSolver_LAPACK : public EigenSolver< SC, LO, GO, NO, OpA, OpB > {

  friend class EigenSolverFactory< SC, LO, GO, NO, OpA, OpB >;

  private:

    using ParameterListPtr = Teuchos::RCP< Teuchos::ParameterList >;
        
    // Constructor
    EigenSolver_LAPACK( OpA opA,
                        OpB opB,
                        const ParameterListPtr parameterList,
                        Teuchos::RCP< std::vector<SC> > &eigenvalues_ptr,
                        Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> >& eigenvectors_ptr,
                        Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> >& choleskyFactor_ptr);

    Teuchos::RCP< std::vector<SC> > solveSymmPosDefGEVP_LAPACK_nondestructive(
          Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> >& matrixA_ptr,
          Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> >& matrixB_ptr,
          const bool computeOnlyEigenvalues = false );

    Teuchos::RCP< std::vector<SC> > solveSymmPosDefGEVP_LAPACK_destructive(
          Teuchos::SerialDenseMatrix<LO,SC>& matrixA,
          Teuchos::SerialDenseMatrix<LO,SC>& matrixB,
          const bool computeOnlyEigenvalues = false );

  };

}

#endif
