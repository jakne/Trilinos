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

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "FROSch_EigenSolver_decl.hpp"

namespace FROSch {

  template < typename SC,   // scalar type
             typename LO,   // local ordinal
             typename GO,   // global ordinal
             typename NO,   // node type [Tpetra::KokkosClassic::DefaultNode::DefaultNodeType]
             typename OpA, 
             typename OpB >
  class EigenSolverFactory {

  public:

      using ParameterListPtr = Teuchos::RCP< Teuchos::ParameterList >;
      using EigenSolverPtr   = Teuchos::RCP< EigenSolver< SC, LO, GO, NO, OpA, OpB > >;

      static EigenSolverPtr Build(OpA opA,
				                          OpB opB,
				                          const ParameterListPtr parameterList,
                                  Teuchos::RCP< std::vector<SC> > &eigenvalues_ptr,
                                  Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> >& eigenvectors_ptr);
  };

}

#endif
