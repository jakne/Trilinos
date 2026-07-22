// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_TYPES_H
#define _FROSCH_TYPES_H

#include <ShyLU_DDFROSch_config.h>
#include <Teuchos_RCPDecl.hpp>

/**
 * @namespace FROSch
 * @brief The FROSch namespace contains the [F]ast and [R]obust [O]verlapping [Sch]warz implementation.
 */
namespace FROSch {

    using Teuchos::RCP;

    #if defined HAVE_TPETRA_INT_INT
    typedef int DefaultGlobalOrdinal;
    #elif !defined HAVE_TPETRA_INT_LONG_LONG
    typedef long DefaultGlobalOrdinal;
    #else
    typedef long long DefaultGlobalOrdinal;
    #endif

    enum DofOrdering {NodeWise=0,DimensionWise=1,Custom=2};

    enum class NullSpaceType
    {
      Laplace = 0,
      Elasticity = 1
    };

    enum Verbosity {None=0,All=1};

    // Types interface components that coarse space basis functions are based on:
    // Vertex functions, edge functions etc.
    enum class InterfaceComponentType {
        Undefined = 0,  // Supposed to always be the default
        Vertex,
        ShortEdge,
        StraightEdge,
        Edge,
        Face,
        Interior
    };

}

#endif
