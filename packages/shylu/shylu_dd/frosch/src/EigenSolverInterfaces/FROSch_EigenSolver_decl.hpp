// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_EIGENSOLVER_DECL_HPP
#define _FROSCH_EIGENSOLVER_DECL_HPP

// Tpetra
//#include <Tpetra_KokkosCompat_DefaultNode.hpp>

//Teuchos
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

// FROSch
#include "FROSch_EigenSolverOperator_decl.hpp"
#include "FROSch_EigenSolverFactory_fwd_decl.hpp"  // forward declaration


namespace FROSch {


template <typename OpA,
          typename OpB,
          typename SC = Tpetra::Details::DefaultTypes::scalar_type, 
          typename LO = Tpetra::Details::DefaultTypes::local_ordinal_type, 
          typename GO = Tpetra::Details::DefaultTypes::global_ordinal_type, 
          typename NO = Tpetra::Details::DefaultTypes::node_type>
class EigenSolver {

friend class EigenSolverFactory< OpA, OpB, SC, LO, GO, NO >;

public:

  using EigenOperatorPtr = Teuchos::RCP< EigenSolverOperator< SC, LO, GO, NO > >;
  using ParameterListPtr = Teuchos::RCP< Teuchos::ParameterList >;

  // Virtual destructor for memory safety of derived classes.
  virtual ~EigenSolver() {};

protected:

  // Constructor
  EigenSolver(OpA opA, OpB opB, ParameterListPtr parameterList, Teuchos::RCP< std::vector<SC> > &eigenvalues_ptr, Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> >& eigenvectors_ptr);

  EigenOperatorPtr opA_;
  EigenOperatorPtr opB_;

  ParameterListPtr parameterList_;

//    bool isInitialized_ = false;

//    bool isComputed_ = false;

private:

  // Default constructer is not allowed. We want to enforce that the EigenSolver is set up by a constructor.
  EigenSolver() = delete;


};


} // end namespace FROSch

#endif
