// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_TOOLS_DECL_HPP
#define _FROSCH_TOOLS_DECL_HPP

#ifdef HAVE_FROSch_DEBUG
#include <chrono>
#include <thread>
#endif

#include <ShyLU_DDFROSch_config.h>

#include <FROSch_Types.h>
#include <FROSch_Output.h>
#include <FROSch_Timers.h>

#include <Tpetra_Distributor.hpp>
#include <Tpetra_FECrsMatrix_decl.hpp>
#include <Tpetra_Assembly_Helpers.hpp>
#include <MatrixMarket_Tpetra.hpp>

#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_CrsGraphFactory.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_ExportFactory.hpp>

#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
#include <Zoltan2_MatrixAdapter.hpp>
#include <Zoltan2_XpetraCrsMatrixAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_XpetraCrsGraphAdapter.hpp>
#endif


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;


    template <typename LO,
              typename GO>
    class OverlappingData {

    protected:

        using IntVec        = Array<int>;

        using LOVec         = Array<LO>;

    public:

        OverlappingData(GO gid,
                        int pid,
                        LO lid);

        int Merge(const RCP<OverlappingData<LO,GO> > od) const;


        GO GID_;

        mutable IntVec PIDs_;

        mutable LOVec LIDs_;

    };

    template <typename LO,typename GO>
    int MergeList(Array<RCP<OverlappingData<LO,GO> > > &odList);

    template <typename LO,
              typename GO,
              typename NO>
    class LowerPIDTieBreak : public Tpetra::Details::TieBreak<LO,GO> {

    protected:

        using CommPtr                   = RCP<const Comm<int> >;

        using XMap                      = Map<LO,GO,NO>;
        using XMapPtr                   = RCP<XMap>;
        using ConstXMapPtr              = RCP<const XMap>;
        using XMapPtrVecPtr             = ArrayRCP<XMapPtr>;
        using ConstXMapPtrVecPtr        = ArrayRCP<ConstXMapPtr>;
        using XMapPtrVecPtr2D           = ArrayRCP<XMapPtrVecPtr>;
        using ConstXMapPtrVecPtr2D      = ArrayRCP<ConstXMapPtrVecPtr>;

        using OverlappingDataPtr        = RCP<OverlappingData<LO,GO> >;
        using OverlappingDataPtrVec     = Array<OverlappingDataPtr>;

        using UN                        = unsigned;

        using IntVec                    = Array<int>;
        using IntVecVecPtr              = ArrayRCP<IntVec>;

        using LOVec                     = Array<LO>;

        using GOVec                     = Array<GO>;
        using GOVecPtr                  = ArrayRCP<GO>;
        using GOVecVec                  = Array<GOVec>;
        using GOVecVecPtr               = ArrayRCP<GOVec>;

        using GOView                    = Kokkos::View<GO*, Kokkos::HostSpace>;

    public:
        LowerPIDTieBreak(CommPtr comm,
                         ConstXMapPtr originalMap,
                         UN dimension,
                         UN levelID = 1); // This is in order to estimate the length of SendImageIDs_ and ExportEntries_ in advance

        virtual bool mayHaveSideEffects() const {
            return false;
        }

        IntVecVecPtr& getComponents()
        {
            return ComponentsSubdomains_;
        }

        int sendDataToOriginalMap();

        virtual size_t selectedIndex(GO GID,
                                          const vector<pair<int,LO> > & pid_and_lid) const;

    protected:

        CommPtr MpiComm_;

        ConstXMapPtr OriginalMap_;

        mutable LO ElementCounter_; // This is mutable such that it can be modified in selectedIndex()

        mutable OverlappingDataPtrVec OverlappingDataList_; // This is mutable such that it can be modified in selectedIndex()

        IntVecVecPtr ComponentsSubdomains_; // This is mutable such that it can be modified in selectedIndex()

        UN LevelID_ = 1;
    };

    template <class SC, class LO, class GO, class NO>
    void writeMM(std::string fileName, Teuchos::RCP<Xpetra::Matrix<SC,LO,GO,NO> > &matrix_);

    template<class SC, class LO, class GO, class NO>
    void writeMM(Teuchos::RCP<Xpetra::Matrix<SC,LO,GO,NO> >& matrix_,std::string fileName);

    template <class SC, class LO, class GO, class NO>
    void readMM(std::string fileName, Teuchos::RCP<Xpetra::Matrix<SC,LO,GO,NO> > &matrix_,RCP<const Comm<int> > &comm);

    template <class SC,class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildRepeatedMapGaleriStruct2D(RCP<const Matrix<SC,LO,GO,NO> > matrix,int M,int Dim);


    template <class SC,class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildRepeatedMapGaleriStruct3D(RCP<const Map<LO,GO,NO> > matrix,int M,int Dim);

    template <class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > BuildUniqueMap(const RCP<const Map<LO,GO,NO> > map,
                                             bool useCreateOneToOneMap = true,
                                             RCP<Tpetra::Details::TieBreak<LO,GO> > tieBreak = null);

    template <class SC,class LO,class GO,class NO>
    ArrayRCP<RCP<const Map<LO,GO,NO> > > BuildRepeatedSubMaps(RCP<const Matrix<SC,LO,GO,NO> > matrix,
                                                              ArrayRCP<const RCP<Map<LO,GO,NO> > > subMaps);

    template <class SC,class LO,class GO,class NO>
    ArrayRCP<RCP<const Map<LO,GO,NO> > > BuildRepeatedSubMaps(RCP<const CrsGraph<LO,GO,NO> > graph,
                                                              ArrayRCP<const RCP<Map<LO,GO,NO> > > subMaps);

    template <class SC,class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildRepeatedMapNonConstOld(RCP<const Matrix<SC,LO,GO,NO> > matrix);

    template <class SC,class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > BuildRepeatedMapOld(RCP<const Matrix<SC,LO,GO,NO> > matrix);

    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildRepeatedMapNonConstOld(RCP<const CrsGraph<LO,GO,NO> > graph);

    template <class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > BuildRepeatedMapOld(RCP<const CrsGraph<LO,GO,NO> > graph);

    template <class SC,class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildRepeatedMapNonConst(RCP<const Matrix<SC,LO,GO,NO> > matrix);

    template <class SC,class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > BuildRepeatedMap(RCP<const Matrix<SC,LO,GO,NO> > matrix);

    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildRepeatedMapNonConst(RCP<const CrsGraph<LO,GO,NO> > graph);

    template <class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > BuildRepeatedMap(RCP<const CrsGraph<LO,GO,NO> > graph);

    template <class LO,class GO,class NO>
    Teuchos::RCP<Xpetra::Map<LO,GO,NO> > BuildMapFromNodeMapRepeated(Teuchos::RCP<const Xpetra::Map<LO,GO,NO> > &nodesMap,
                                                                     unsigned dofsPerNode,
                                                                     unsigned dofOrdering);

    template <class SC,class LO,class GO,class NO>
    int ExtendOverlapByOneLayer_Old(RCP<const Matrix<SC,LO,GO,NO> > inputMatrix,
                                    RCP<const Map<LO,GO,NO> > inputMap,
                                    RCP<const Matrix<SC,LO,GO,NO> > &outputMatrix,
                                    RCP<const Map<LO,GO,NO> > &outputMap);

    template <class SC,class LO,class GO,class NO>
    int ExtendOverlapByOneLayer(RCP<const Matrix<SC,LO,GO,NO> > inputMatrix,
                                RCP<const Map<LO,GO,NO> > inputMap,
                                RCP<const Matrix<SC,LO,GO,NO> > &outputMatrix,
                                RCP<const Map<LO,GO,NO> > &outputMap);

    template <class LO,class GO,class NO>
    int ExtendOverlapByOneLayer(RCP<const CrsGraph<LO,GO,NO> > inputGraph,
                                RCP<const Map<LO,GO,NO> > inputMap,
                                RCP<const CrsGraph<LO,GO,NO> > &outputGraph,
                                RCP<const Map<LO,GO,NO> > &outputMap);

    /*! \brief Sort the Xpetra::Map by the global IDs \c x
     * \param[in] inputMap Unsorted input map
     */
    template <class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > SortMapByGlobalIndex(RCP<const Map<LO,GO,NO> > inputMap);

    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > AssembleMaps(ArrayView<RCP<const Map<LO,GO,NO> > > mapVector,
                                     ArrayRCP<ArrayRCP<LO> > &partMappings);

    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > AssembleMapsNonConst(ArrayView<RCP<Map<LO,GO,NO> > > mapVector,
                                             ArrayRCP<ArrayRCP<LO> > &partMappings);

    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > AssembleSubdomainMap(unsigned numberOfBlocks,
                                             ArrayRCP<ArrayRCP<RCP<const Map<LO,GO,NO> > > > dofsMaps,
                                             ArrayRCP<unsigned> dofsPerNode);

    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > MergeMapsNonConst(ArrayRCP<RCP<const Map<LO,GO,NO> > > mapVector);

    template <class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > MergeMaps(ArrayRCP<RCP<const Map<LO,GO,NO> > > mapVector);

    template <class LO,class GO,class NO>
    int BuildDofMaps(const RCP<const Map<LO,GO,NO> > map,
                     unsigned dofsPerNode,
                     unsigned dofOrdering,
                     RCP<const Map<LO,GO,NO> > &nodesMap,
                     ArrayRCP<RCP<const Map<LO,GO,NO> > > &dofMaps,
                     GO offset = 0);

    template <class LO,class GO,class NO>
    int BuildDofMapsVec(const ArrayRCP<RCP<const Map<LO,GO,NO> > > mapVec,
                        ArrayRCP<unsigned> dofsPerNodeVec,
                        ArrayRCP<FROSch::DofOrdering> dofOrderingVec,
                        ArrayRCP<RCP<const Map<LO,GO,NO> > > &nodesMapVec,
                        ArrayRCP<ArrayRCP<RCP<const Map<LO,GO,NO> > > >&dofMapsVec);


    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildMapFromDofMaps(const ArrayRCP<RCP<Map<LO,GO,NO> > > &dofMaps,
                                            unsigned dofsPerNode,
                                            unsigned dofOrdering);

    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildMapFromNodeMap(RCP<const Map<LO,GO,NO> > &nodesMap,
                                            unsigned dofsPerNode,
                                            unsigned dofOrdering);

    template <class LO,class GO,class NO>
    ArrayRCP<RCP<const Map<LO,GO,NO> > > BuildNodeMapsFromDofMaps(ArrayRCP<ArrayRCP<RCP<const Map<LO,GO,NO> > > >dofsMapsVecVec,
                                                                  ArrayRCP<unsigned> dofsPerNodeVec,
                                                                  ArrayRCP<DofOrdering> dofOrderingVec);

    template <class LO,class GO,class NO>
    ArrayRCP<RCP<Map<LO,GO,NO> > > BuildSubMaps(RCP<const Map<LO,GO,NO> > &fullMap,
                                                ArrayRCP<GO> maxSubGIDVec);

    template <class SC,class LO,class GO,class NO>
    ArrayRCP<GO> FindOneEntryOnlyRowsGlobal(RCP<const Matrix<SC,LO,GO,NO> > matrix,
                                            RCP<const Map<LO,GO,NO> > repeatedMap,
                                            typename ScalarTraits<SC>::magnitudeType tol = 0.0);

    template <class LO,class GO,class NO>
    ArrayRCP<GO> FindOneEntryOnlyRowsGlobal(RCP<const CrsGraph<LO,GO,NO> > graph,
                                            RCP<const Map<LO,GO,NO> > repeatedMap);

    template <class SC,class LO>
    bool ismultiple(ArrayView<SC> A,
                    ArrayView<SC> B);

    template<class T>
    inline void sort(T &v);

    template<class T>
    inline void sortunique(T &v);

    template <class SC, class LO,class GO,class NO>
    RCP<MultiVector<SC,LO,GO,NO> > ModifiedGramSchmidt(RCP<const MultiVector<SC,LO,GO,NO> > multiVector,
                                                       ArrayView<unsigned> zero = ArrayView<unsigned>());

    template <class SC, class LO,class GO,class NO>
    RCP<const MultiVector<SC,LO,GO,NO> > BuildNullSpace(unsigned dimension,
                                                        const NullSpaceType nullSpaceType,
                                                        RCP<const Map<LO,GO,NO> > repeatedMap,
                                                        unsigned dofsPerNode,
                                                        ArrayRCP<RCP<const Map<LO,GO,NO> > > dofsMaps,
                                                        RCP<const MultiVector<SC,LO,GO,NO> > nodeList = null);

    template <class Type>
    RCP<Type> ExtractPtrFromParameterList(ParameterList& paramList,
                                          string namePtr="Ptr");

    template <class Type>
    ArrayRCP<Type> ExtractVectorFromParameterList(ParameterList& paramList,
                                                  string nameVector="Vector");

    template <class LO>
    Array<LO> GetIndicesFromString(string string);

#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
    template <class SC,class LO,class GO,class NO>
    int RepartionMatrixZoltan2(RCP<Matrix<SC,LO,GO,NO> > &crsMatrix,
                               RCP<ParameterList> parameterList);

    template <class LO,class GO, class NO>
    int BuildRepMapZoltan(RCP<CrsGraph<LO,GO,NO> > Xgraph,
                          RCP<CrsGraph<LO,GO,NO> > B,
                          RCP<ParameterList> parameterList,
                          Teuchos::RCP<const Teuchos::Comm<int> > TeuchosComm,
                          RCP<Map<LO,GO,NO> > &RepeatedMap);
#endif

    template <typename SC, typename LO, typename GO, typename NO>
    Teuchos::RCP<Tpetra::FECrsMatrix<SC,LO,GO,NO>> FECrsMatrix_deepCopy(const Teuchos::RCP<const Tpetra::FECrsMatrix<SC,LO,GO,NO>> matrix_in, Teuchos::RCP<Tpetra::FECrsGraph<>> fe_graph);

    template <typename SC, typename LO, typename GO, typename NO> 
    Teuchos::RCP<Teuchos::SerialDenseMatrix<LO,SC>> convert_LocalSquareXMatrix_to_SerialDenseMatrix(Teuchos::RCP<const Xpetra::Matrix<SC,LO,GO,NO>> matrix_in);

#ifdef HAVE_FROSch_DEBUG
    namespace debug {
        /*!
        \brief Print map to terminal.
        
        Example: FROSch::debug::printMap(map, "Row Map", __FILE__, __LINE__);

        Sample output:
        Row map of stiffness matrix (local index | global index)  [/home/peter/trilinos/src/packages/shylu/shylu_dd/frosch/test/example/main.cpp:103]
        rank (0): (0|0) (1|1) (2|2) (3|7) (4|8) (5|9) (6|14) (7|15) (8|16) 
        rank (1): (0|3) (1|4) (2|5) (3|6) (4|10) (5|11) (6|12) (7|13) (8|17) (9|18) (10|19) (11|20) 
        rank (2): (0|21) (1|22) (2|23) (3|28) (4|29) (5|30) (6|35) (7|36) (8|37) (9|42) (10|43) (11|44) 
        rank (3): (0|24) (1|25) (2|26) (3|27) (4|31) (5|32) (6|33) (7|34) (8|38) (9|39) (10|40) (11|41) (12|45) (13|46) (14|47) (15|48) 

        \param[in] map Example: RCP<const Xpetra::Map<LO, GO, NO>> or RCP<Tpetra::Map<LO, GO, NO>>
        \param[in] strInfoToDisplay A string printed out to the console
        \param[in] file Filename where this function is called from: __FILE__
        \param[in] line Line number where this function is called from: __LINE__
        */
        template <typename map_type>
        void printMap(const Teuchos::RCP<map_type> map, const std::string strInfoToDisplay = "", const char* file = 0, const int line = -1);

        /*!
        \brief Print Xpetra matrix to terminal, based on what each rank holds, in dense format, i.e., including zeros.
        
        Example: FROSch::debug::printXpetraMatrix_local_dense(matrix, "Stiffness matrix", __FILE__, __LINE__);

        Sample output:
        Stiffness matrix  [/home/peter/trilinos/src/packages/shylu/shylu_dd/frosch/test/example/main.cpp:103]
        global num rows: 49
        global num cols: 49
        rank(0) local num rows: 9
        rank(0) local num cols: 16
        1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  
        0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  
        0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  
        0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  
        0  0  0  0  668.667  -166.833  0  -166.833  -333.333  0  0  0  0  0  0  0  
        0  0  0  0  -166.833  668.667  0  -333.333  -166.833  0  -0.333333  -0.333333  0  0  0  0  
        0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  
        0  0  0  0  -166.833  -333.333  0  1334.67  -333.333  0  0  0  0  -166.833  -333.333  0  
        0  0  0  0  -333.333  -166.833  0  -333.333  1334.67  0  -0.333333  -0.333333  0  -333.333  -166.833  -0.333333  
        rank(1) local num rows: 12
        rank(1) local num cols: 20
            *** not showing output ***
        rank(2) local num rows: 12
        rank(2) local num cols: 20
            *** not showing output ***
        rank(3) local num rows: 16
        rank(3) local num cols: 25
            *** not showing output ***

        \param[in] M matrix. Example: RCP<const Xpetra::Matrix<SC, LO, GO, NO>> or RCP<Xpetra::Matrix<SC, LO, GO, NO>>. Tpetra::Matrix is not supported.
        \param[in] strInfoToDisplay A string printed out to the console
        \param[in] file Filename where this function is called from: __FILE__
        \param[in] line Line number where this function is called from: __LINE__
        */
        template <typename matrix_type>
        void printXpetraMatrix_local_dense(const Teuchos::RCP<matrix_type> M, const std::string strInfoToDisplay = "", const char* file = 0, const int line = -1);

        /*!
        \brief Print Tpetra matrix to terminal, based on what each rank holds, in sparse format.
        
        Example: FROSch::debug::printTpetraCrsMatrix_local_sparse(matrix, "Stiffness matrix", __FILE__, __LINE__);

        Sample output:
        Row map of stiffness matrix (local index | global index)  [/home/peter/trilinos/src/packages/shylu/shylu_dd/frosch/test/example/main.cpp:103]

        Stiffness matrix  [/home/peter/trilinos/src/packages/shylu/shylu_dd/frosch/test/example/main.cpp:103]
        global num rows: 49
        global num cols: 49
        rank(0) local num rows: 9
        rank(0) local num cols: 16
        rank(0) matrix entries: [local row id | global row id] (local column id | global column id | value)
           [0|0]  (0|0|1)
           [1|1]  (1|1|2)
           [2|2]  (2|2|2)
           [3|7]  (3|7|2)
           [4|8]  (4|8|668.667) (5|9|-166.833) (7|15|-166.833) (8|16|-333.333) 
           [5|9]  (4|8|-166.833) (5|9|668.667) (7|15|-333.333) (8|16|-166.833) (10|10|-0.333333) (11|17|-0.333333) 
           [6|14]  (6|14|2)
           [7|15]  (4|8|-166.833) (5|9|-333.333) (7|15|1334.67) (8|16|-333.333) (13|22|-166.833) (14|23|-333.333) 
           [8|16]  (4|8|-333.333) (5|9|-166.833) (7|15|-333.333) (8|16|1334.67) (10|10|-0.333333) (11|17|-0.333333) (13|22|-333.333) (14|23|-166.833) (15|24|-0.333333) 
        rank(1) local num rows: 12
        rank(1) local num cols: 20
        rank(1) matrix entries: [local row id | global row id] (local column id | global column id | value)
            *** not showing output ***
        rank(2) : local num rows: 12
        rank(2) : local num cols: 20
        rank(2) matrix entries: [local row id | global row id] (local column id | global column id | value)
            *** not showing output ***
        rank(3) : local num rows: 16
        rank(3) : local num cols: 25
        rank(3) matrix entries: [local row id | global row id] (local column id | global column id | value)
            *** not showing output ***

        \param[in] M matrix. Example: RCP of Tpetra::FECrsMatrix<SC,LO,GO,NO> or Tpetra::CrsMatrix<SC,LO,GO,NO>, either const or nonconst. FECrsMatrix can be printed before endAssembly (fillComplete). Xpetra::Matrix is not supported.
        \param[in] strInfoToDisplay A string printed out to the console
        \param[in] file Filename where this function is called from: __FILE__
        \param[in] line Line number where this function is called from: __LINE__
        */
        template <typename matrix_type>
        void printTpetraCrsMatrix_local_sparse(const Teuchos::RCP<matrix_type> M, const std::string strInfoToDisplay = "", const char* file = 0, const int line = -1);

        // TODO: missing info
        template <typename matrix_type>
        void printXpetraMatrix_local_sparse(const Teuchos::RCP<matrix_type> M, const std::string strInfoToDisplay = "", const char* file = 0, const int line = -1);
    }
#endif

    /*!
    \brief Throw runtime error due to missing package in build configuration

    As many packages are optional, we might detect only at runtime that a certain package
    is not included in the build configuration but is used by FROSch.
    Use this routine to throw a generic error message with some information for the user
    and provide details how to fix it.

    \param[in] froschObj FROSch object that is asking for the missing package
    \param[in] packageName Name of the missing package
    */
    inline void ThrowErrorMissingPackage(const string& froschObj,
                                         const string& packageName)
    {
        // Create the error message
        stringstream errMsg;
        errMsg << froschObj << " is asking for the Trilinos package '"<< packageName << "', "
        "but this package is not included in your build configuration. "
        "Please enable '" << packageName << "' in your build configuration to be used with ShyLU_DDFROSch.";

        // Throw the error
        FROSCH_ASSERT(false, errMsg.str());
    }
}

#endif
