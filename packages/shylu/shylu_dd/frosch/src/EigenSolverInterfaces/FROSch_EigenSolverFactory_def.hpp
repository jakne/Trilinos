// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_EIGENSOLVER_FACTORY_DEF_HPP
#define _FROSCH_EIGENSOLVER_FACTORY_DEF_HPP

#include <ShyLU_DDFROSch_config.h>
// TODO: FROSch_Types and Tools use Xpetra namespace.
namespace Xpetra {
}
#include <FROSch_Tools_decl.hpp>
#include <FROSch_Types.h>

#include "FROSch_EigenSolverFactory_decl.hpp"

// Eigen solvers
#include "FROSch_EigenSolver_LAPACK_def.hpp"
// #ifdef HAVE_SHYLU_DDFROSCH_ANASAZI

namespace FROSch {

template <typename OpA, typename OpB, typename SC, typename LO, typename GO, typename NO>
typename EigenSolverFactory<OpA, OpB, SC, LO, GO, NO>::EigenSolverPtr
EigenSolverFactory<OpA, OpB, SC, LO, GO, NO>::Solve(OpA opA,
                                                    OpB opB,
                                                    const ParameterListPtr parameterList,
                                                    Teuchos::RCP<std::vector<SC>> &eigenvalues_ptr,
                                                    Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &eigenvectors_ptr)
{
    EigenSolverFactory<OpA, OpB, SC, LO, GO, NO>::EigenSolverPtr solverPtr;

    // Per default, we use LAPACK as a solver.
    const string solverType = parameterList->get("Solver Type", "LAPACK");
    if (!solverType.compare("LAPACK")) {
        const ParameterListPtr sublistLAPACK =
            Teuchos::sublist(Teuchos::sublist(parameterList, "Solver Types"), "LAPACK");
        Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> choleskyFactor_ptr;
        solverPtr = Teuchos::rcp(new EigenSolver_LAPACK<OpA, OpB, SC, LO, GO, NO>(
            opA, opB, sublistLAPACK, eigenvalues_ptr, eigenvectors_ptr, choleskyFactor_ptr));
    } else if (!solverType.compare("LOBPCG")) {
#ifdef HAVE_SHYLU_DDFROSCH_ANASAZI
        FROSCH_ASSERT(false,
                      "FROSch::EigenSolverFactory: Iterative eigensolvers have "
                      "not been integrated.");
#else
        ThrowErrorMissingPackage("FROSch::EigenSolverFactory", "Anasazi");
#endif
    } else {
        FROSCH_ASSERT(false, "FROSch::EigenSolverFactory:Build   Unknown solver type [" + solverType + "].");
    }
    return solverPtr;
}
} // namespace FROSch

#endif
