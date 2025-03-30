// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_EIGENSOLVER_OPERATOR_DEF_HPP
#define _FROSCH_EIGENSOLVER_OPERATOR_DEF_HPP

#include <FROSch_Output.h>
#include "FROSch_EigenSolverOperator_decl.hpp"


namespace FROSch {

/*
template< typename SC, typename LO, typename GO, typename NO >
bool EigenSolverOperator<SC,LO,GO,NO>::isInitialized() const
{
    return isInitialized_;
}

template< typename SC, typename LO, typename GO, typename NO >
bool EigenSolverOperator<SC,LO,GO,NO>::isComputed() const
{
    return isComputed_;
}
*/

template < typename SC, typename LO, typename GO, typename NO >
EigenSolverOperator< SC, LO, GO, NO >::EigenSolverOperator( const bool isFunction ) : 
  matrix_MV_ptr_( NULL ), matrix_Crs_ptr_( NULL ), matrix_Dense_ptr_( NULL ), isFunction_( isFunction )
{
  if (!isFunction) {
	std::stringstream errMsg;
	errMsg << "Error. "
                "FROSch::EigenSolverOperator::EigenSolverOperator(isFunction)"
                ": isFunction must be true. Otherwise the operator would be uninitialized. "
                "This is not allowed.";
    FROSCH_ASSERT(false, errMsg.str());
  }
}

template < typename SC, typename LO, typename GO, typename NO >
const bool EigenSolverOperator< SC, LO, GO, NO >::isSerialDenseMatrix() const {
  return (!matrix_Dense_ptr_.is_null());
}

} // end namespace FROSch

#endif
