// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_EXTRACTSUBMATRICES_DECL_HPP
#define _FROSCH_EXTRACTSUBMATRICES_DECL_HPP

#include <ShyLU_DDFROSch_config.h>

#include <FROSch_Types.h>
#include <FROSch_Timers.h>

#include <Xpetra_MapFactory_fwd.hpp>
#include <Xpetra_MatrixFactory_fwd.hpp>
#include <Xpetra_Matrix_decl.hpp>
#include <Xpetra_ImportFactory_fwd.hpp>
#include <Xpetra_Map_decl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Tpetra_FECrsMatrix_decl.hpp>

#include <KokkosKernels_Utils.hpp>


namespace FROSch {

    template <typename SC, typename LO, typename GO, typename NO>
    typename Teuchos::RCP<const Xpetra::Matrix<SC,LO,GO,NO> > ExtractLocalSubdomainMatrix_feTest(
            Teuchos::RCP< const Tpetra::FECrsMatrix<SC,LO,GO,NO> > globalMatrix,
            Teuchos::RCP< const Xpetra::Map<LO,GO,NO> >            map);

    template <class SC,class LO,class GO,class NO>
    RCP<const Xpetra::Matrix<SC,LO,GO,NO> > ExtractLocalSubdomainMatrix(
            RCP<const Xpetra::Matrix<SC,LO,GO,NO> > globalMatrix,
            RCP<const Xpetra::Map<LO,GO,NO> > map);

    // ----------------------------------------------------------- //
    // split ExtractLocalSubdomainMatrix into symbolic / compute
    template <class SC,class LO,class GO,class NO>
    void ExtractLocalSubdomainMatrix_Symbolic(RCP<Xpetra::Matrix<SC,LO,GO,NO> > subdomainMatrix,        // input  : globalMatrix, re-distributed with map
                                              RCP<Xpetra::Matrix<SC,LO,GO,NO> > localSubdomainMatrix);  // output : local submatrix

    template <class SC,class LO,class GO,class NO>
    void ExtractLocalSubdomainMatrix_Compute(RCP<const Xpetra::Matrix<SC,LO,GO,NO> > globalMatrix,
                                             RCP<      Xpetra::Matrix<SC,LO,GO,NO> > subdomainMatrix,
                                             RCP<      Xpetra::Matrix<SC,LO,GO,NO> > repeatedMatrix);

    template <class SC,class LO,class GO,class NO>
    void ExtractLocalSubdomainMatrix_Compute(RCP<      Xpetra::Import<LO,GO,NO> >    scatter,
                                             RCP<const Xpetra::Matrix<SC,LO,GO,NO> > globalMatrix,
                                             RCP<      Xpetra::Matrix<SC,LO,GO,NO> > subdomainMatrix,
                                             RCP<      Xpetra::Matrix<SC,LO,GO,NO> > repeatedMatrix);
    // ----------------------------------------------------------- //

    template <class SC,class LO,class GO,class NO>
    RCP<const Xpetra::Matrix<SC,LO,GO,NO> > ExtractLocalSubdomainMatrix(
            RCP<const Xpetra::Matrix<SC,LO,GO,NO> > globalMatrix,
            RCP<const Xpetra::Map<LO,GO,NO> >       map,
            SC                                      value);

    template <class SC,class LO,class GO,class NO>
    int UpdateLocalSubdomainMatrix(RCP<Xpetra::Matrix<SC,LO,GO,NO> > globalMatrix,
                                   RCP<Xpetra::Map<LO,GO,NO> >       &map,
                                   RCP<Xpetra::Matrix<SC,LO,GO,NO> > &localSubdomainMatrix);

    template <class SC,class LO,class GO,class NO>
    int BuildSubmatrices(RCP<const Xpetra::Matrix<SC,LO,GO,NO> > k,
                         Teuchos::ArrayView<GO>                  indI,
                         RCP<const Xpetra::Matrix<SC,LO,GO,NO> > &kII,
                         RCP<const Xpetra::Matrix<SC,LO,GO,NO> > &kIJ,
                         RCP<const Xpetra::Matrix<SC,LO,GO,NO> > &kJI,
                         RCP<const Xpetra::Matrix<SC,LO,GO,NO> > &kJJ);

    template <class SC,class LO,class GO,class NO>
    int BuildSubmatrix(RCP<const Xpetra::Matrix<SC,LO,GO,NO> > k,
                       Teuchos::ArrayView<GO>                  indI,
                       RCP<const Xpetra::Matrix<SC,LO,GO,NO> > &kII);

    template <class LO,class GO,class NO>
    int BuildSubgraph(RCP<const Xpetra::CrsGraph<LO,GO,NO> > k,
                      Teuchos::ArrayView<GO>                 indI,
                      RCP<const Xpetra::CrsGraph<LO,GO,NO> > &kII);
}

#endif
