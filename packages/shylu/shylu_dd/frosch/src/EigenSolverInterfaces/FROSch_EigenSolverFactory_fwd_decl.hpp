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

#include <Tpetra_Details_DefaultTypes.hpp>

namespace FROSch {

template <typename OpA, 
          typename OpB, 
          typename SC = Tpetra::Details::DefaultTypes::scalar_type, 
          typename LO = Tpetra::Details::DefaultTypes::local_ordinal_type,
          typename GO = Tpetra::Details::DefaultTypes::global_ordinal_type, 
          typename NO = Tpetra::Details::DefaultTypes::node_type>
class EigenSolverFactory;

}

#endif
