// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_EIGENSOLVER_OPERATOR_SCHUR_COMPLEMENT_DECL_HPP
#define _FROSCH_EIGENSOLVER_OPERATOR_SCHUR_COMPLEMENT_DECL_HPP

#include <ShyLU_DDFROSch_config.h>

// FROSch
#include <FROSch_EigenSolver_def.hpp>

#include "FROSch_EigenSolverFactory_fwd_decl.hpp"  // forward declaration


namespace FROSch {

    /**
     * @brief 
     */
    template < typename SC = double,
               typename LO = int,
               typename GO = DefaultGlobalOrdinal,
               typename NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType >
    class EigenSolverOperator_SchurComplement : public EigenSolverOperator< SC, LO, GO, NO > {

    friend class EigenSolverFactory< SC, LO, GO, NO >;

//    protected:

        // Teuchos
//        using ParameterListPtr                  = typename Solver<SC,LO,GO,NO>::ParameterListPtr;

/*
    public:

        //! Initialize the internal solver
        virtual int initialize();

        //! Compute the internal solver
        virtual int compute();
*/

    private:
        
        // Constructor
        EigenSolver_LAPACK(EigenOperatorPtr opA, EigenOperatorPtr opB, ParameterListPtr parameterList);

    };

}

#endif
