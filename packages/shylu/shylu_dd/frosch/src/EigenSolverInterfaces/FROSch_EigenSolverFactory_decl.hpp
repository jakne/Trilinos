// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_EIGENSOLVER_FACTORY_DECL_HPP
#define _FROSCH_EIGENSOLVER_FACTORY_DECL_HPP

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include "FROSch_EigenSolver_decl.hpp"

namespace FROSch {

template <typename OpA, typename OpB, typename SC, typename LO, typename GO, typename NO>
class EigenSolverFactory {

  public:
    using ParameterListPtr = Teuchos::RCP<Teuchos::ParameterList>;
    using EigenSolverPtr = Teuchos::RCP<EigenSolver<OpA, OpB, SC, LO, GO, NO>>;

    static EigenSolverPtr Build(OpA opA,
                                OpB opB,
                                const ParameterListPtr parameterList,
                                Teuchos::RCP<std::vector<SC>> &eigenvalues_ptr,
                                Teuchos::RCP<Teuchos::SerialDenseMatrix<LO, SC>> &eigenvectors_ptr);
};

} // namespace FROSch

#endif
