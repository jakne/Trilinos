#ifndef _FROSCH_EIGENSOLVER_FACTORY_DEF_HPP
#define _FROSCH_EIGENSOLVER_FACTORY_DEF_HPP

#include <ShyLU_DDFROSch_config.h>
namespace Xpetra {}  // TODO: FROSch_Types and Tools use Xpetra namespace.
#include <FROSch_Types.h>
#include <FROSch_Tools_decl.hpp>

#include "FROSch_EigenSolverFactory_decl.hpp"

// Eigen solvers
#include "FROSch_EigenSolver_LAPACK_def.hpp"
//#ifdef HAVE_SHYLU_DDFROSCH_ANASAZI


namespace FROSch {

  // static function
  template < typename SC, typename LO, typename GO, typename NO, typename OpA, typename OpB >
  typename EigenSolverFactory< SC, LO, GO, NO, OpA, OpB >::EigenSolverPtr 
  EigenSolverFactory< SC, LO, GO, NO, OpA, OpB >::Build(OpA opA,
							                                          OpB opB,
							                                          const ParameterListPtr parameterList,
                                                        Teuchos::RCP< std::vector<SC> > &eigenvalues_ptr,
                                                        Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> >& eigenvectors_ptr)
  {
    EigenSolverFactory< SC, LO, GO, NO, OpA, OpB >::EigenSolverPtr solverPtr;

    // Per default, we use LAPACK as a solver.
    const string solverType = parameterList->get("Solver Type","LAPACK");
    if (!solverType.compare("LAPACK")) {
      const ParameterListPtr sublistLAPACK = Teuchos::sublist(Teuchos::sublist(parameterList,"Solver Types"),"LAPACK");
      Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> > choleskyFactor_ptr;
      solverPtr = Teuchos::rcp(new EigenSolver_LAPACK< SC, LO, GO, NO, OpA, OpB >( opA, opB, sublistLAPACK, eigenvalues_ptr, eigenvectors_ptr, choleskyFactor_ptr ));
    } else if (!solverType.compare("LOBPCG")) {
      FROSCH_ASSERT(false, "FROSch::EigenSolverFactory: Iterative eigensolvers have not been integrated.");
#ifdef HAVE_SHYLU_DDFROSCH_ANASAZI
#else
      ThrowErrorMissingPackage("FROSch::EigenSolverFactory","Anasazi");
#endif
    } else {
      FROSCH_ASSERT(false, "FROSch::EigenSolverFactory:Build   Unknown solver type [" + solverType + "].");
    }
    return solverPtr;
  }
}

#endif
