// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef FROSCH_EIGENSOLVER_OPERATOR_DECL_HPP
#define FROSCH_EIGENSOLVER_OPERATOR_DECL_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <FROSch_Output.h>

namespace FROSch {

/**
 * @class EigenSolverOperator
 * @brief Class for operators used in the generalized eigenvalue problem. 
 *
 * @details
 * The generalized eigenvalue problem A*x = lambda*B*x supports [$$$ does it?] 
 * the operators A and B to be Crs matrices, serial dense matrices, multivectors, 
 * and functions that implement the application of x.
 * The latter could, for example, be the application of a Schur complement and 
 * needs to be implemented by the deriving class.
 * The operators should be used for a call to the EigenSolverFactory:
 *    EigenSolverFactory.Build(of type EigenSolverOperator, 
 *                             of type EigenSolverOperator, 
 *                             options)
 * 
 * @tparam SC   scalar type
 * @tparam LO   local ordinal type
 * @tparam GO   global ordinal type; @see FROSch::DefaultGlobalOrdinal
 * @tparam NO   node type; @see Tpetra::KokkosClassic::DefaultNode::DefaultNodeType
*/

template < typename SC, typename LO, typename GO, typename NO >
class EigenSolverOperator {

public:

  EigenSolverOperator( const Teuchos::RCP< Tpetra::MultiVector< SC, LO, GO > > matrix_MV_ptr ) : 
    matrix_MV_ptr_( matrix_MV_ptr ), matrix_Crs_ptr_( NULL ), matrix_Dense_ptr_( NULL ), isFunction_( false ) {}

  EigenSolverOperator( const Teuchos::RCP< Tpetra::CrsMatrix< SC, LO, GO > > matrix_Crs_ptr ) : 
    matrix_MV_ptr_( NULL ), matrix_Crs_ptr_( matrix_Crs_ptr ), matrix_Dense_ptr_( NULL ), isFunction_( false ) {}

  EigenSolverOperator( Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > > matrix_Dense_ptr ) : 
    matrix_MV_ptr_( NULL ), matrix_Crs_ptr_( NULL ), matrix_Dense_ptr_( matrix_Dense_ptr ), isFunction_( false ) {}

  // Virtual destructor for memory safety of derived classes.
  virtual ~EigenSolverOperator() {};

  /**
   * @brief Apply
  */
  virtual void
  apply ( const Tpetra::MultiVector< SC, LO, GO >& X,
          Tpetra::MultiVector< SC, LO, GO >& Y ) const {}

  const bool isSerialDenseMatrix() const;
  Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > > getSerialDenseMatrixPtr() {return matrix_Dense_ptr_;} // $$$

protected:

  EigenSolverOperator( const bool isFunction );

private:
  // Default constructer is not allowed. We want to enforce that exactly one type of operator is 
  // passed as an argument.
  EigenSolverOperator() = delete;

  // There are 4 options to store the matrix A.
  // 1) is multivector. TODO: It most likely must always be converted to a matrix.
  // 2--3) are matrices, sparse and dense.
  // 4) is a function that implements the application of A*x. 
  //    In this case, the derived class needs to implement apply(X,Y) and store the 
  //    required variables.
  // The derived classes $$$ will need to do what exactly about the conversion of matrices.
  // $$$ So far, we will only support the dense matrices.
  const Teuchos::RCP< Tpetra::MultiVector< SC, LO, GO > > matrix_MV_ptr_;
  const Teuchos::RCP< Tpetra::CrsMatrix< SC, LO, GO > > matrix_Crs_ptr_;
  Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > > matrix_Dense_ptr_;  // $$$ const? problem ...
  const bool isFunction_;

};

} // end FROSch namespace

#endif
