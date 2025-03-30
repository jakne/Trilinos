// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_EIGENSOLVER_FACTORY_FWD_DECL_HPP
#define _FROSCH_EIGENSOLVER_FACTORY_FWD_DECL_HPP

#include <Tpetra_KokkosCompat_DefaultNode.hpp>

namespace FROSch {

template < typename SC, // = double,                // scalar type
           typename LO, // = int,                   // local ordinal
           typename GO, // = DefaultGlobalOrdinal,  // global ordinal
           typename NO, // = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType >
	       typename OpA, 
	       typename OpB >
class EigenSolverFactory;

}

#endif
