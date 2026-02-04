// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_GDSWCOARSEOPERATOR_DEF_HPP
#define _FROSCH_GDSWCOARSEOPERATOR_DEF_HPP

#include <iostream>
#include <chrono>


#include <FROSch_GDSWCoarseOperator_decl.hpp>

#include <FROSch_EigenSolver_def.hpp>
#include <FROSch_EigenSolverFactory_def.hpp>
#include <Tpetra_FECrsMatrix.hpp>

namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    GDSWCoarseOperator<SC,LO,GO,NO>::GDSWCoarseOperator(ConstXMatrixPtr k,
                                                        ParameterListPtr parameterList) :
    HarmonicCoarseOperator<SC,LO,GO,NO> (k,parameterList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(gDSWCoarseOperatorTime,"GDSWCoarseOperator::GDSWCoarseOperator");
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::initialize(UN dimension,
                                                    ConstXMapPtr repeatedMap)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"GDSWCoarseOperator::initialize");
        buildCoarseSpace(dimension,repeatedMap);
        this->assembleInterfaceCoarseSpace();
        this->buildCoarseSolveMap(this->AssembledInterfaceCoarseSpace_->getBasisMapUnique());
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::initialize(UN dimension,
                                                    ConstXMapPtr repeatedMap,
                                                    GOVecPtr dirichletBoundaryDofs)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"GDSWCoarseOperator::initialize");
        buildCoarseSpace(dimension,repeatedMap,dirichletBoundaryDofs);
        this->assembleInterfaceCoarseSpace();
        this->buildCoarseSolveMap(this->AssembledInterfaceCoarseSpace_->getBasisMapUnique());
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::initialize(UN dimension,
                                                    UN dofsPerNode,
                                                    ConstXMapPtr repeatedNodesMap,
                                                    ConstXMapPtrVecPtr repeatedDofMaps)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"GDSWCoarseOperator::initialize");
        buildCoarseSpace(dimension,dofsPerNode,repeatedNodesMap,repeatedDofMaps);
        this->assembleInterfaceCoarseSpace();
        this->buildCoarseSolveMap(this->AssembledInterfaceCoarseSpace_->getBasisMapUnique());
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::initialize(UN dimension,
                                                    UN dofsPerNode,
                                                    ConstXMapPtr repeatedNodesMap,
                                                    ConstXMapPtrVecPtr repeatedDofMaps,
                                                    GOVecPtr dirichletBoundaryDofs)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"GDSWCoarseOperator::initialize");
        buildCoarseSpace(dimension,dofsPerNode,repeatedNodesMap,repeatedDofMaps,dirichletBoundaryDofs);
        this->assembleInterfaceCoarseSpace();
        this->buildCoarseSolveMap(this->AssembledInterfaceCoarseSpace_->getBasisMapUnique());
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::initialize(UN dimension,
                                                    UN dofsPerNode,
                                                    ConstXMapPtr repeatedNodesMap,
                                                    ConstXMapPtrVecPtr repeatedDofMaps,
                                                    ConstXMultiVectorPtr nodeList)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"GDSWCoarseOperator::initialize");
        buildCoarseSpace(dimension,dofsPerNode,repeatedNodesMap,repeatedDofMaps,nodeList);
        this->assembleInterfaceCoarseSpace();
        this->buildCoarseSolveMap(this->AssembledInterfaceCoarseSpace_->getBasisMapUnique());
        this->extractLocalSubdomainMatrix_Symbolic();
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::initialize(UN dimension,
                                                    UN dofsPerNode,
                                                    ConstXMapPtr repeatedNodesMap,
                                                    ConstXMapPtrVecPtr repeatedDofMaps,
                                                    GOVecPtr dirichletBoundaryDofs,
                                                    ConstXMultiVectorPtr nodeList)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"GDSWCoarseOperator::initialize");
        buildCoarseSpace(dimension,dofsPerNode,repeatedNodesMap,repeatedDofMaps,dirichletBoundaryDofs,nodeList);
        this->assembleInterfaceCoarseSpace();
        this->buildCoarseSolveMap(this->AssembledInterfaceCoarseSpace_->getBasisMapUnique());
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::initialize(UN dimension,
                                                    UNVecPtr dofsPerNodeVec,
                                                    ConstXMapPtrVecPtr repeatedNodesMapVec,
                                                    ConstXMapPtrVecPtr2D repeatedDofMapsVec,
                                                    GOVecPtr2D dirichletBoundaryDofsVec,
                                                    ConstXMultiVectorPtrVecPtr nodeListVec)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"GDSWCoarseOperator::initialize");
        buildCoarseSpace(dimension,dofsPerNodeVec,repeatedNodesMapVec,repeatedDofMapsVec,dirichletBoundaryDofsVec,nodeListVec);
        this->assembleInterfaceCoarseSpace();
        this->buildCoarseSolveMap(this->AssembledInterfaceCoarseSpace_->getBasisMapUnique());
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    void GDSWCoarseOperator<SC,LO,GO,NO>::describe(FancyOStream &out,
                                                   const EVerbosityLevel verbLevel) const
    {
        FROSCH_ASSERT(false,"describe() has to be implemented properly...");
    }

    template <class SC,class LO,class GO,class NO>
    string GDSWCoarseOperator<SC,LO,GO,NO>::description() const
    {
        return "GDSW Coarse Operator";
    }

    template<class SC,class LO,class GO,class NO>
    typename GDSWCoarseOperator<SC,LO,GO,NO>::XMapPtr GDSWCoarseOperator<SC,LO,GO,NO>::BuildRepeatedMapCoarseLevel(ConstXMapPtr &nodesMap,
                                                UN dofsPerNode,
                                                ConstXMapPtrVecPtr dofsMaps,
                                                UN partitionType)
    {
      FROSCH_ASSERT(false,"For GDSWCoarseOperator the ZoltanDual Option is not implemented!");
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::buildCoarseSpace(UN dimension,
                                                          ConstXMapPtr nodesMap)
    {
        ConstXMapPtrVecPtr dofsMaps(1);
        dofsMaps[0] = nodesMap;
        buildCoarseSpace(dimension,1,nodesMap,dofsMaps);

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::buildCoarseSpace(UN dimension,
                                                          ConstXMapPtr nodesMap,
                                                          GOVecPtr dirichletBoundaryDofs)
    {
        ConstXMapPtrVecPtr dofsMaps(1);
        dofsMaps[0] = nodesMap;
        buildCoarseSpace(dimension,1,nodesMap,dofsMaps,dirichletBoundaryDofs);

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::buildCoarseSpace(UN dimension,
                                                          UN dofsPerNode,
                                                          ConstXMapPtr nodesMap,
                                                          ConstXMapPtrVecPtr dofsMaps)
    {
/*
#ifdef FindOneEntryOnlyRowsGlobal_Matrix
        GOVecPtr dirichletBoundaryDofs = FindOneEntryOnlyRowsGlobal(this->K_.getConst(),nodesMap);
#else
        GOVecPtr dirichletBoundaryDofs = FindOneEntryOnlyRowsGlobal(this->K_->getCrsGraph(),nodesMap);
#end
 */
        FROSCH_WARNING("FROSch::GDSWCoarseOperator",this->Verbose_,"We do not have the right map (repeatedMap) to use FindOneEntryOnlyRowsGlobal. A variant that uses the row map could be implemented?! => We use dirichletBoundaryDofs = null for now.");
        GOVecPtr dirichletBoundaryDofs = null;
        buildCoarseSpace(dimension,dofsPerNode,nodesMap,dofsMaps,dirichletBoundaryDofs);

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::buildCoarseSpace(UN dimension,
                                                          UN dofsPerNode,
                                                          ConstXMapPtr nodesMap,
                                                          ConstXMapPtrVecPtr dofsMaps,
                                                          GOVecPtr dirichletBoundaryDofs)
    {
        ConstXMultiVectorPtr nodeList;
        buildCoarseSpace(dimension,dofsPerNode,nodesMap,dofsMaps,dirichletBoundaryDofs,nodeList);

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::buildCoarseSpace(UN dimension,
                                                          UN dofsPerNode,
                                                          ConstXMapPtr nodesMap,
                                                          ConstXMapPtrVecPtr dofsMaps,
                                                          ConstXMultiVectorPtr nodeList)
    {

// TODO: [JK] This and dependent parts need to be fixed and properly implemented. The coarse functions should be zero on the global Dirichlet boundary.
//#ifdef FindOneEntryOnlyRowsGlobal_Matrix
        GOVecPtr dirichletBoundaryDofs = FindOneEntryOnlyRowsGlobal(this->K_.getConst(),nodesMap);
//#else
//        GOVecPtr dirichletBoundaryDofs = FindOneEntryOnlyRowsGlobal(this->K_->getCrsGraph(),nodesMap);
//#end

        FROSCH_WARNING("FROSch::GDSWCoarseOperator",this->Verbose_,"We do not have the right map (repeatedMap) to use FindOneEntryOnlyRowsGlobal. A variant that uses the row map could be implemented?! => We use dirichletBoundaryDofs = null for now.");
//        GOVecPtr dirichletBoundaryDofs = null;
        buildCoarseSpace(dimension,dofsPerNode,nodesMap,dofsMaps,dirichletBoundaryDofs,nodeList);

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::buildCoarseSpace(UN dimension,
                                                          UN dofsPerNode,
                                                          ConstXMapPtr nodesMap,
                                                          ConstXMapPtrVecPtr dofsMaps,
                                                          GOVecPtr dirichletBoundaryDofs,
                                                          ConstXMultiVectorPtr nodeList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(buildCoarseSpaceTime,"GDSWCoarseOperator::buildCoarseSpace");
        FROSCH_ASSERT(dofsMaps.size()==dofsPerNode,"dofsMaps.size()!=dofsPerNode");

        // Das könnte man noch ändern
        // TODO: DAS SOLLTE ALLES IN EINE FUNKTION IN HARMONICCOARSEOPERATOR
        resetCoarseSpaceBlock(this->NumberOfBlocks_,dimension,dofsPerNode,nodesMap,dofsMaps,dirichletBoundaryDofs,nodeList);

        return 0;
    }



    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::buildCoarseSpace(UN dimension,
                                                          UNVecPtr dofsPerNodeVec,
                                                          ConstXMapPtrVecPtr repeatedNodesMapVec,
                                                          ConstXMapPtrVecPtr2D repeatedDofMapsVec,
                                                          GOVecPtr2D dirichletBoundaryDofsVec,
                                                          ConstXMultiVectorPtrVecPtr nodeListVec)
    {
        FROSCH_DETAILTIMER_START_LEVELID(buildCoarseSpaceTime,"GDSWCoarseOperator::buildCoarseSpace");
        // Das könnte man noch ändern
        // TODO: DAS SOLLTE ALLES IN EINE FUNKTION IN HARMONICCOARSEOPERATOR
        for (UN i=0; i<repeatedNodesMapVec.size(); i++) {
            resetCoarseSpaceBlock(this->NumberOfBlocks_,dimension,dofsPerNodeVec[i],repeatedNodesMapVec[i],repeatedDofMapsVec[i],dirichletBoundaryDofsVec[i],nodeListVec[i]);
        }
        return 0;
    }


    template <class SC,class LO,class GO,class NO>
    int GDSWCoarseOperator<SC,LO,GO,NO>::resetCoarseSpaceBlock(UN blockId,
                                                               UN dimension,
                                                               UN dofsPerNode,
                                                               ConstXMapPtr nodesMap,
                                                               ConstXMapPtrVecPtr dofsMaps,
                                                               GOVecPtr dirichletBoundaryDofs,
                                                               ConstXMultiVectorPtr nodeList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(resetCoarseSpaceBlockTime,"GDSWCoarseOperator::resetCoarseSpaceBlock");
        FROSCH_ASSERT(dofsMaps.size()==dofsPerNode,"dofsMaps.size()!=dofsPerNode");
        FROSCH_ASSERT(blockId<=this->NumberOfBlocks_,"Block does not exist yet and can therefore not be reset("+to_string(blockId)+" <= "+to_string(this->NumberOfBlocks_)+". ");

        if (!this->DistributionList_->get("Type","linear").compare("ZoltanDual")) {
            FROSCH_ASSERT(false,"RGDSWCoarseOperator:: Distribution Type ZoltanDual only works for IPOUHarmonicCoarseOperator");
        }

        // Process the parameter list
        stringstream blockIdStringstream;
        blockIdStringstream << blockId+1;
        string blockIdString = blockIdStringstream.str();
        RCP<ParameterList> coarseSpaceList = sublist(sublist(this->ParameterList_,"Blocks"),blockIdString.c_str());

        CommunicationStrategy communicationStrategy = CreateOneToOneMap;
        if (!coarseSpaceList->get("Interface Communication Strategy","CreateOneToOneMap").compare("CrsMatrix")) {
            communicationStrategy = CommCrsMatrix;
        } else if (!coarseSpaceList->get("Interface Communication Strategy","CreateOneToOneMap").compare("CrsGraph")) {
            communicationStrategy = CommCrsGraph;
        } else if (!coarseSpaceList->get("Interface Communication Strategy","CreateOneToOneMap").compare("CreateOneToOneMap")) {
            communicationStrategy = CreateOneToOneMap;
        } else {
            FROSCH_ASSERT(false,"FROSch::GDSWCoarseOperator: Specify a valid communication strategy for the identification of the interface components.");
        }

        Verbosity verbosity = All;
        if (!coarseSpaceList->get("Verbosity","All").compare("None")) {
            verbosity = None;
        } else if (!coarseSpaceList->get("Verbosity","All").compare("All")) {
            verbosity = All;
        } else {
            FROSCH_ASSERT(false,"FROSch::GDSWCoarseOperator: Specify a valid verbosity level.");
        }

        bool useForCoarseSpace = coarseSpaceList->get("Use For Coarse Space",true);

        bool useVertexTranslations = coarseSpaceList->sublist("Custom").get("Vertices: translations",true);

        bool useShortEdgeTranslations = coarseSpaceList->sublist("Custom").get("ShortEdges: translations",true);
        bool useShortEdgeRotations = coarseSpaceList->sublist("Custom").get("ShortEdges: rotations",true);

        bool useStraightEdgeTranslations = coarseSpaceList->sublist("Custom").get("StraightEdges: translations",true);
        bool useStraightEdgeRotations = coarseSpaceList->sublist("Custom").get("StraightEdges: rotations",true);

        bool useEdgeTranslations = coarseSpaceList->sublist("Custom").get("Edges: translations",true);
        bool useEdgeRotations = coarseSpaceList->sublist("Custom").get("Edges: rotations",true);

        bool useFaceTranslations = coarseSpaceList->sublist("Custom").get("Faces: translations",true);
        bool useFaceRotations = coarseSpaceList->sublist("Custom").get("Faces: rotations",true);

        bool useRotations = coarseSpaceList->get("Rotations",true);
        if (useRotations && nodeList.is_null()) {
            useRotations = false;
            FROSCH_WARNING("FROSch::GDSWCoarseOperator",this->Verbose_,"Rotations cannot be used since nodeList.is_null().");
        }
        if (!useRotations) {
            useShortEdgeRotations = false;
            useStraightEdgeRotations = false;
            useEdgeRotations = false;
            useFaceRotations = false;
        }

        if (useForCoarseSpace) {
            this->NumberOfBlocks_++;

            this->GammaDofs_.resize(this->GammaDofs_.size()+1);
            this->IDofs_.resize(this->IDofs_.size()+1);
            this->InterfaceCoarseSpaces_.resize(this->InterfaceCoarseSpaces_.size()+1);
            this->DofsMaps_.resize(this->DofsMaps_.size()+1);
            this->DofsPerNode_.resize(this->DofsPerNode_.size()+1);

            this->DofsMaps_[blockId] = dofsMaps;
            this->DofsPerNode_[blockId] = dofsPerNode;

            Array<GO> tmpDirichletBoundaryDofs(dirichletBoundaryDofs()); // Here, we do a copy. Maybe, this is not necessary
            sortunique(tmpDirichletBoundaryDofs);

            DDInterface_.reset(new DDInterface<SC,LO,GO,NO>(dimension,this->DofsPerNode_[blockId],nodesMap.getConst(),verbosity,this->LevelID_,communicationStrategy));
            DDInterface_->resetGlobalDofs(dofsMaps);
            DDInterface_->removeDirichletNodes(tmpDirichletBoundaryDofs());

            if (useVertexTranslations||useShortEdgeTranslations||useShortEdgeRotations||useStraightEdgeTranslations||useStraightEdgeRotations||useEdgeTranslations||useEdgeRotations||useFaceTranslations||useFaceRotations) {
                EntitySetPtr interface = this->DDInterface_->getInterface();
                EntitySetPtr interior = this->DDInterface_->getInterior();

                if (this->Verbose_) {
                    cout
                    << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                    << setw(89) << "-----------------------------------------------------------------------------------------"
                    << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                    << "| "
                    << left << setw(74) << "GDSWCoarseOperator " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")"
                    << " |"
                    << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                    << setw(89) << "========================================================================================="
                    << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                    << "| " << left << setw(41) << "Block" << right
                    << " | " << setw(41) << blockId
                    << " |"
                    << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                    << "| " << left << setw(41) << "Spatial dimensions" << right
                    << " | " << setw(41) << dimension
                    << " |"
                    << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                    << "| " << left << setw(41) << "Number of degrees of freedom per node" << right
                    << " | " << setw(41) << dofsPerNode
                    << " |"
                    << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                    << setw(89) << "-----------------------------------------------------------------------------------------"
                    << endl;
                }

                // Check for interface
                if (interface->getEntity(0)->getNumNodes()==0) {
                    // getEntity(0): There is only one interface, thus, only one entity.
                    FROSCH_NOTIFICATION("FROSch::GDSWCoarseOperator",this->Verbose_,"No interface found => Volume functions will be used instead.");
                    this->computeVolumeFunctions(blockId,dimension,nodesMap,nodeList,interior);
                } else {
                    this->GammaDofs_[blockId] = LOVecPtr(this->DofsPerNode_[blockId]*interface->getEntity(0)->getNumNodes());
                    this->IDofs_[blockId] = LOVecPtr(this->DofsPerNode_[blockId]*interior->getEntity(0)->getNumNodes());
                    for (UN k=0; k<this->DofsPerNode_[blockId]; k++) {
                        for (UN i=0; i<interface->getEntity(0)->getNumNodes(); i++) {
                            this->GammaDofs_[blockId][this->DofsPerNode_[blockId]*i+k] = interface->getEntity(0)->getLocalDofID(i,k);
                        }
                        for (UN i=0; i<interior->getEntity(0)->getNumNodes(); i++) {
                            this->IDofs_[blockId][this->DofsPerNode_[blockId]*i+k] = interior->getEntity(0)->getLocalDofID(i,k);
                        }
                    }

                    this->InterfaceCoarseSpaces_[blockId].reset(new CoarseSpace<SC,LO,GO,NO>(this->MpiComm_,this->SerialComm_));

                    if (this->ParameterList_->get("Test Unconnected Interface",true)) {
                        DDInterface_->divideUnconnectedEntities(this->K_);
                    }

                    // Classify interface components: Which nodes are vertices, shortEdges etc.
                    DDInterface_->sortVerticesEdgesFaces(nodeList);

                    // EntitySetPtr interface = DDInterface_->getInterface();
                    // EntitySetPtr interior = DDInterface_->getInterior();

                    ////////////////////////////////
                    // Build Processor Map Coarse //
                    ////////////////////////////////
                    DDInterface_->buildEntityMaps(useVertexTranslations,
                                                  useShortEdgeTranslations||useShortEdgeRotations,
                                                  useStraightEdgeTranslations || useStraightEdgeRotations,
                                                  useEdgeTranslations || useEdgeRotations,
                                                  useFaceTranslations || useFaceRotations,
                                                  false);

                    // Vertices
                    if (useVertexTranslations) {
                        XMultiVectorPtrVecPtr translations = this->computeTranslations(blockId,DDInterface_->getVertices());
                        ConstXMapPtr verticesEntityMap = DDInterface_->getVertices()->getEntityMap();
                        for (UN i=0; i<translations.size(); i++) {
                            this->InterfaceCoarseSpaces_[blockId]->addSubspace(verticesEntityMap,null,translations[i]);
                        }
                    }
                    // ShortEdges
                    if (useShortEdgeTranslations) {
                        XMultiVectorPtrVecPtr translations = this->computeTranslations(blockId,DDInterface_->getShortEdges());
                        ConstXMapPtr shortEdgesEntityMap = DDInterface_->getShortEdges()->getEntityMap();
                        for (UN i=0; i<translations.size(); i++) {
                            this->InterfaceCoarseSpaces_[blockId]->addSubspace(shortEdgesEntityMap,null,translations[i]);
                        }
                    }
                    if (useShortEdgeRotations) {
                        XMultiVectorPtrVecPtr rotations = this->computeRotations(blockId,dimension,nodeList,DDInterface_->getShortEdges(),(dimension==3));
                        ConstXMapPtr shortEdgesEntityMap = DDInterface_->getShortEdges()->getEntityMap();
                        for (UN i=0; i<rotations.size(); i++) {
                            this->InterfaceCoarseSpaces_[blockId]->addSubspace(shortEdgesEntityMap,null,rotations[i]);
                        }
                    }
                    // StraightEdges
                    if (useStraightEdgeTranslations) {
                        XMultiVectorPtrVecPtr translations = this->computeTranslations(blockId,DDInterface_->getStraightEdges());
                        ConstXMapPtr straightEdgesEntityMap = DDInterface_->getStraightEdges()->getEntityMap();
                        for (UN i=0; i<translations.size(); i++) {
                            this->InterfaceCoarseSpaces_[blockId]->addSubspace(straightEdgesEntityMap,null,translations[i]);
                        }
                    }
                    if (useStraightEdgeRotations) {
                        XMultiVectorPtrVecPtr rotations = this->computeRotations(blockId,dimension,nodeList,DDInterface_->getStraightEdges(),(dimension==3));
                        ConstXMapPtr straightEdgesEntityMap = DDInterface_->getStraightEdges()->getEntityMap();
                        for (UN i=0; i<rotations.size(); i++) {
                            this->InterfaceCoarseSpaces_[blockId]->addSubspace(straightEdgesEntityMap,null,rotations[i]);
                        }
                    }
                    // Edges
                    if (useEdgeTranslations) {
                        XMultiVectorPtrVecPtr translations = this->computeTranslations(blockId,DDInterface_->getEdges());
                        ConstXMapPtr edgesEntityMap = DDInterface_->getEdges()->getEntityMap();
                        for (UN i=0; i<translations.size(); i++) {
                            this->InterfaceCoarseSpaces_[blockId]->addSubspace(edgesEntityMap,null,translations[i]);
                        }
                    }
                    if (useEdgeRotations) {
                        XMultiVectorPtrVecPtr rotations = this->computeRotations(blockId,dimension,nodeList,DDInterface_->getEdges());
                        ConstXMapPtr edgesEntityMap = DDInterface_->getEdges()->getEntityMap();
                        for (UN i=0; i<rotations.size(); i++) {
                            this->InterfaceCoarseSpaces_[blockId]->addSubspace(edgesEntityMap,null,rotations[i]);
                        }
                    }

                    bool useAdaptiveCoarseSpace = this->ParameterList_->get("Use Adaptive Coarse Space",true);

                    // Faces
                    if (useFaceTranslations & !useAdaptiveCoarseSpace) {
                        FROSCH_TIMER_START_LEVELID(timeFacesGDSW,"GDSWCoarseOperator::resetCoarseSpaceBlock::GDSW face functions");
                        XMultiVectorPtrVecPtr translations = this->computeTranslations(blockId,DDInterface_->getFaces());
                        ConstXMapPtr facesEntityMap = DDInterface_->getFaces()->getEntityMap();
                        for (UN i=0; i<translations.size(); i++) {
                            this->InterfaceCoarseSpaces_[blockId]->addSubspace(facesEntityMap,null,translations[i]);
                        }
                        FROSCH_TIMER_STOP(timeFacesGDSW);


//                        this->MpiComm_->barrier();
//                        std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
//                        std::cout << translations[0]->getNumVectors() << "  " << translations[0]->getLocalLength() << "  " << translations[0]->getGlobalLength() << std::endl;
//                        this->MpiComm_->barrier();
//                        std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
//
//                        FROSch::debug::printMap(facesEntityMap,"facesEntityMap_",__FILE__,__LINE__);

                    } else if (useFaceTranslations & useAdaptiveCoarseSpace) {
                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions");

//                        ConstXMapPtr facesEntityMap_ = DDInterface_->getFaces()->getEntityMap();
//                        FROSch::debug::printMap(facesEntityMap_,"facesEntityMap_",__FILE__,__LINE__);
		    

                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW1,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (1)");
                        Teuchos::RCP<Teuchos::ParameterList> parameterList_adaptiveProblems = Teuchos::sublist(this->ParameterList_, "Adaptive problems");
                        const bool addMPIBarriersForSomeTimers = parameterList_adaptiveProblems->get("Add MPI barriers for some timers", true);

                        const int numFaces_global = (int)DDInterface_->getFaces()->getEntityMap()->getMaxAllGlobalIndex()+1; // +1 since indices are zero based
                        XMultiVectorPtrVecPtr translations = XMultiVectorPtrVecPtr(1);  // TODO: rename variable translations
//                        XMultiVectorPtrVecPtr translations = XMultiVectorPtrVecPtr(numFaces_global);  // TODO: rename variable translations

                        this->ParameterList_->print();

                        Teuchos::RCP< Tpetra::FECrsMatrix<SC,LO,GO,NO> > dummy = Teuchos::null;
                        Teuchos::RCP< Tpetra::FECrsMatrix<SC,LO,GO,NO> > fe_matrix = this->ParameterList_->get("Neumann Matrices",dummy);
                        FROSCH_TIMER_STOP(timeFacesAGDSW1);

                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW2,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (2) Extract subdomain matrix (repeated)");
                        ConstXMapPtr repeatedMap;
                        ConstXMatrixPtr repeatedMatrix;
                        repeatedMap = FROSch::AssembleSubdomainMap(this->NumberOfBlocks_,this->DofsMaps_,this->DofsPerNode_);
                        if (fe_matrix == Teuchos::null) {
                            repeatedMatrix = FROSch::ExtractLocalSubdomainMatrix(this->K_,repeatedMap.getConst());
                        } else {
                            repeatedMatrix = FROSch::ExtractLocalSubdomainMatrix(fe_matrix.getConst(),repeatedMap.getConst());
                        }
                        FROSCH_TIMER_STOP(timeFacesAGDSW2);

                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW3,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (3)");
                        // List of all local nodes that is used later to extract submatrices
                        const int numEl = nodesMap->getLocalNumElements();
                        std::vector<GO> allLocalNodes(numEl); // not dofs! needs to be changed.
                        for (int ii = 0; ii < numEl; ii++) {
                            allLocalNodes[ii] = ii;
                        }

                        // Map that is completely local and contains all interface DOFs of the subdomain.
                        // Later on, when the map is used to store the eigenfunctions on the items (e.g., edges), another map is used to map from the item to the local interface.
                        XMapPtr serialGammaMap = MapFactory<LO,GO,NO>::Build(this->K_->getRowMap()->lib(),this->GammaDofs_[0].size(),0,this->SerialComm_);
                        //FROSch::debug::printMap(serialGammaMap,"serialGammaMap",__FILE__,__LINE__);
                        FROSCH_TIMER_STOP(timeFacesAGDSW3);

                        // Set up sub communicators.
                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW4,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (4): set up sub communicators");
                        Teuchos::Array<Teuchos::RCP< const Teuchos::Comm<int> >> subcomms(0);
                        using InterfaceEntityPtr = typename SchwarzOperator<SC,LO,GO,NO>::InterfaceEntityPtr;
                        MPI_Group world_group;
                        // TODO: Implement as in Teuchos test reduceAllInPlace.cpp (getRawMpiCommFromTeuchosComm)
                        const Teuchos::MpiComm<int>* mpiCommPtr = dynamic_cast<const Teuchos::MpiComm<int>* > (this->MpiComm_.get());
                        Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> > wrapper = mpiCommPtr->getRawMpiComm();
                        MPI_Comm rawMpiComm = *wrapper;
                        MPI_Comm_group(rawMpiComm, &world_group);
                        // Since we iterate over the global ID of a face, faces locally are then sorted by their global ID.
                        // Since we iterater over 0 to numFaces_global-1, the global IDs are also sorted.
                        // Later, this means, that the local IDs are sorted by the values of the global IDs.
                        // As a result: If subdomain 1 processes its local edge 1 first, 
                        // this will also be the first edge of the neighboring subdomain if they share this edge.
                        // This prevents a deadlock situation, but it is not efficient, since any subdomain waits for previous
                        // subdomains to finish, before doing work with their neighbors who also wait for other subdomains.
                        // In a square that is subdivided into square subdomains, the subdomains at the top right could start
                        // work independently of the subdomains in the lower left.
                        for (int ii = 0; ii < numFaces_global; ii++){
                            const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid();
                            const LO localEntityID = DDInterface_->getFaces()->getEntityMap()->getLocalElement(ii);

                            // Create communicator for the subdomains neighboring the entity.
                            const bool isSubdomainNeighborOfEntity = (localEntityID != INVALID);

                            if (isSubdomainNeighborOfEntity) {

                                const InterfaceEntityPtr entity_ptr = DDInterface_->getFaces()->getEntity(localEntityID);

                                const typename SchwarzOperator<SC,LO,GO,NO>::IntVec subdomainsVector = entity_ptr->getSubdomainsVector();
                                
                                // std::cout << "subdomainsVector" << std::endl;
                                // commNeighborsOfEntity->barrier();
                                // std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
                                // for (int ww = 0; ww < subdomainsVector.size(); ww++) {
                                //     std::cout << commNeighborsOfEntity->getRank() << " | " << subdomainsVector[ww] << std::endl;
                                //     commNeighborsOfEntity->barrier();
                                //     std::this_thread::sleep_for(std::chrono::nanoseconds(5000));
                                // }
                                // commNeighborsOfEntity->barrier();
                                // std::this_thread::sleep_for(std::chrono::nanoseconds(50000));

                                // std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
                                int* ranks_to_include = new int[subdomainsVector.size()];
                                for (int ww = 0; ww < subdomainsVector.size(); ww++) {
                                    ranks_to_include[ww] = subdomainsVector[ww];
                                    // std::cout << this->MpiComm_->getRank() << " | " << subdomainsVector[ww] << std::endl;
                                }
                                // std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
                                int num_ranks = subdomainsVector.size();

                                MPI_Group new_group;
                                MPI_Group_incl(world_group, num_ranks, ranks_to_include, &new_group);
                                MPI_Comm new_comm_;
                                MPI_Comm_create_group(rawMpiComm, new_group, ii, &new_comm_);
                                Teuchos::RCP<const Teuchos::Comm<int>> commNeighborsOfEntity = Teuchos::rcp(new Teuchos::MpiComm<int>(new_comm_));

                                delete[] ranks_to_include;

                                subcomms.push_back(commNeighborsOfEntity);
                            }
                        }
                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();

                        // Within the neighboorhood communicator, the rank with ID 0 shall be the root.
                        // It will be that rank that will compute the sum of local Schur complements and that will solve the eigenvalue problem.
                        const int rootRankOfNeighborhood = 0;

                        FROSCH_TIMER_STOP(timeFacesAGDSW4);

                        // local-to-global map of local face IDs
                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW5,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (5): local-to-global map of local face IDs");
                        //const int numFaces_local = DDInterface_->getFaces()->getNumEntities();
                        int numFacesLocal = 0;
                        Teuchos::Array<GO> globalFaceIDsOfSubdomain(0);
                        Teuchos::Array<LO> localEntityIDsOfSubdomain(0);
                        for (int ii = 0; ii < numFaces_global; ii++){ // See comment at the last loop over the global faces
                            const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid();
                            const LO localEntityID = DDInterface_->getFaces()->getEntityMap()->getLocalElement(ii);
                            if (localEntityID != INVALID) {
                                globalFaceIDsOfSubdomain.push_back(ii);
                                localEntityIDsOfSubdomain.push_back(localEntityID);
                                numFacesLocal += 1;
                            }
                        }
                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();
                        FROSCH_TIMER_STOP(timeFacesAGDSW5);

                        // Create maps for the DOFs of an item (e.g., of an edge)
                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW6,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (6): maps for DOFs of item");
                        Teuchos::Array< Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> > itemMapsRepeated(0), itemMapsUnique(0);
                        Teuchos::Array< Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> > s_ee__MV__unique__list(0),   k_ee__MV__unique__list;
                        Teuchos::Array< Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> > s_ee__MV__repeated__list(0), k_ee__MV__repeated__list(0);
                        Teuchos::Array< std::vector<GO> > collection_itemNodes(0);
                        for (int localFaceID = 0; localFaceID < numFacesLocal; localFaceID++) {
                            LO localEntityID = localEntityIDsOfSubdomain.at(localFaceID);
                            const InterfaceEntityPtr entity_ptr = DDInterface_->getFaces()->getEntity(localEntityID);
                            int numFaceNodes = entity_ptr->getNumNodes();
                            std::vector<GO> itemNodes(0);//numFaceNodes); // not dofs! needs to be changed.
                            itemNodes.resize(numFaceNodes);

                            std::vector<GO> itemNodesGlobalRepeated, itemNodesGlobalUnique;
                            // Get entity nodes.
                            for (int jj = 0; jj < numFaceNodes; jj++) {
                                itemNodes[jj] = entity_ptr->getNode(jj).NodeIDLocal_;
                                itemNodesGlobalRepeated.push_back(entity_ptr->getNode(jj).NodeIDGlobal_);
                            }
                            collection_itemNodes.push_back(itemNodes);

                            Teuchos::RCP< const Teuchos::Comm<int> > commNeighborsOfEntity = subcomms.at(localFaceID);

                            if (commNeighborsOfEntity->getRank() == rootRankOfNeighborhood) {
                                // We select the rank with the smallest ID to do the work (here, we only take the sum of contributions of the local Schur complements: 
                                // S^(i1) + S^(i2) + .... 
                                // For a face in 3D and an edge in 2D, there are only two adjacent subdomains, so the sub communicator only holds the ranks 0 and 1.
                                // This approach may be too simple and a more elaborate load-balancing strategy might be needed.
                                // Satisfied conditions:
                                // * Different nodes/DOFs of an interface component (edge, face, vertex) have the same set of adjacent subdomains
                                //   [would not be true for RAGDSW]
                                // * As a result, all local Schur complements S^(i_j) are (in the case of AGDSW) defined on the same DOFs
                                // Approach:
                                // * We want "sub rank" 0 to have access to all S^(i_j) of the interface component
                                // * Then, "sub rank" 0 can take the sum of all S^(i_j) to obtain a Schur complement corresponding to the interface component.
                                // * The remaining ranks of the sub communicator should hold no DOFs at all.
                                itemNodesGlobalUnique = itemNodesGlobalRepeated;
                            } else {
                                itemNodesGlobalUnique = {};
                            }

                            // TODO: The following call is blocking. Does it matter time-wise and should be improved (re-ordering of the loop indices) or can we ignore that?
                            // Create a map using the list of repeated, global indices (of item DOFs).
                            Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> globalRepeatedMapForItem =
                                Teuchos::rcp(new Tpetra::Map<LO, GO, NO>(
                                    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), // Let Tpetra infer global size
                                    itemNodesGlobalRepeated,  // global indices on this rank
                                    0,          // index base
                                    commNeighborsOfEntity));
                            itemMapsRepeated.push_back(globalRepeatedMapForItem);

                            // TODO: The following call is blocking. Does it matter time-wise and should be improved (re-ordering of the loop indices) or can we ignore that?
                            // Create a map using the list of unique, global indices (of item DOFs).
                            // All indices should be held by only one rank.
                            Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> globalUniqueMapForItem =
                                Teuchos::rcp(new Tpetra::Map<LO, GO, NO>(
                                    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), // Let Tpetra infer global size
                                    itemNodesGlobalUnique,  // global indices on this rank
                                    0,          // index base
                                    commNeighborsOfEntity));
                            itemMapsUnique.push_back(globalUniqueMapForItem);

                            // Although globalUniqueMapForItem and globalRepeatedMapForItem hold global indices, during the 
                            // creation of the MultiVector and exporter/importer, these will be mapped to some contiguous
                            // index set. For example: (1,9,88) could be mapped to (0,1,2). As a result, the size of following 
                            // MultiVectors equals the number of unique indices.

                            // The values s_ee__MV->getNumVectors(), globalUniqueMapForItem->getGlobalNumElements(), numFaceNodes should all coincide. TODO: numFaceNodes should become numFaceDOFs, but nothing else should change.
                            // Create neighbor-global, distributed, unique MultiVector. The rank with lowest ID will hold all data. The remaining ones don't hold any data.
                            // Also create a corresponding repeated MultiVector.

                            // TODO: The following four calls are blocking. Does it matter time-wise and should be improved (re-ordering of the loop indices) or can we ignore that?
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> s_ee__MV__unique = Teuchos::rcp(new Tpetra::MultiVector<SC, LO, GO, NO>(globalUniqueMapForItem, numFaceNodes));
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> k_ee__MV__unique = Teuchos::rcp(new Tpetra::MultiVector<SC, LO, GO, NO>(globalUniqueMapForItem, numFaceNodes));
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> s_ee__MV__repeated = Teuchos::rcp(new Tpetra::MultiVector<SC, LO, GO, NO>(globalRepeatedMapForItem, numFaceNodes));
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> k_ee__MV__repeated = Teuchos::rcp(new Tpetra::MultiVector<SC, LO, GO, NO>(globalRepeatedMapForItem, numFaceNodes));

                            s_ee__MV__repeated__list.push_back(s_ee__MV__repeated);
                            k_ee__MV__repeated__list.push_back(k_ee__MV__repeated);
                            s_ee__MV__unique__list.push_back(s_ee__MV__unique);
                            k_ee__MV__unique__list.push_back(k_ee__MV__unique);
                        }
                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();
                        FROSCH_TIMER_STOP(timeFacesAGDSW6);

                        // Compute local contributions to eigenvalue problem (no communication).
                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW7_loop,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (7): loop over local faces");
                        Teuchos::Array<XMultiVectorPtr> evpRHSs(0);
                        Teuchos::Array<XMultiVectorPtr> evpLHSs(0);
                        for (int localFaceID = 0; localFaceID < numFacesLocal; localFaceID++) {
                            FROSCH_TIMER_START_LEVELID(timeFacesAGDSW7_loop_1,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (7): loop [1]");
                            LO localEntityID = localEntityIDsOfSubdomain.at(localFaceID);
                            std::vector<GO> itemNodes = collection_itemNodes.at(localFaceID);

                            // get number of face nodes
                            const InterfaceEntityPtr entity_ptr = DDInterface_->getFaces()->getEntity(localEntityID);
                            int numFaceNodes = entity_ptr->getNumNodes(); // not dofs! needs to be changed.
                            // int maxNumFaceNodes_ranks = 0;
                            // reduceAll(*this->MpiComm_,Teuchos::REDUCE_MAX,numFaceNodes,ptr(&maxNumFaceNodes_ranks));
                            // numFaceNodes = maxNumFaceNodes_ranks;

                            GOVec indicesR(0);  // R:[r]emaining nodes

                            // Fetch split communicator to those subdomains neighboring the entity and the remaining ones.
                            //Teuchos::RCP< const Teuchos::Comm<int> > commNeighborsOfEntity = subcomms.at(localFaceID);

                            // reduceAll(*this->MpiComm_,REDUCE_SUM,localVec[0],ptr(&sumVec[0]));
                            // int minRankIDofCommunicator = 0; // always zero for sub communicator.
                            // reduceAll(*commNeighborsOfEntity,Teuchos::REDUCE_MIN,commNeighborsOfEntity->getRank(),ptr(&maxNumFaceNodes_ranks));

                            // Get set of remaining subdomain nodes.
                            std::vector<int> diff;
                            std::set_difference(allLocalNodes.begin(), allLocalNodes.end(), itemNodes.begin(), itemNodes.end(), std::inserter(diff, diff.begin()));
                            for (int jj = 0; jj < (int)diff.size(); jj++) {
                                indicesR.push_back(diff[jj]);
                            }
                            FROSCH_TIMER_STOP(timeFacesAGDSW7_loop_1);

                            FROSCH_TIMER_START_LEVELID(timeFacesAGDSW7_loop_2,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (7): loop [2]");
                            XMatrixPtr k_ee, k_RR, k_Re, k_eR;
                            FROSch::BuildSubmatrices(repeatedMatrix.getConst(),indicesR(),k_RR,k_Re,k_eR,k_ee);
                            FROSCH_TIMER_STOP(timeFacesAGDSW7_loop_2);

                            FROSCH_TIMER_START_LEVELID(timeFacesAGDSW7_loop_3,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (7): loop [3]");
                            // FROSch::debug::printMap(k_ee->getRowMap(),"kee row map",__FILE__,__LINE__);
                            // FROSch::debug::printMap(globalRepeatedMapForItem,"item repeated map",__FILE__,__LINE__);
                            // FROSch::debug::printMap(globalUniqueMapForItem,"item unique map",__FILE__,__LINE__);

                            // [JK] Todo: assert: numFaceNodes == k_ee->getRowMap()->getLocalNumElements()
                            XMultiVectorPtr id_e = MultiVectorFactory<SC,LO,GO,NO>::Build(k_ee->getRowMap(),numFaceNodes);
                            for (int jj=0; jj<numFaceNodes; jj++) {
                                id_e->replaceLocalValue(jj,jj,ScalarTraits<SC>::one());
                            }

                            XMultiVectorPtr k_Re__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(k_Re->getRowMap(),numFaceNodes);
                            k_Re->apply( *id_e, *k_Re__MV );  // (*input,*solution)
                            FROSCH_TIMER_STOP(timeFacesAGDSW7_loop_3);

                            FROSCH_TIMER_START_LEVELID(timeFacesAGDSW7_loop_4,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (7): loop [4]");
                            // Solve k_RR * X = k_Re__MV.
                            // --> inv_k_RR__k_Re__MV := X.
                            XMultiVectorPtr inv_k_RR__k_Re__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(k_RR->getRowMap(),numFaceNodes);
                            this->ExtensionSolver_ = SolverFactory<SC,LO,GO,NO>::Build(k_RR,
                                                                 sublist(this->ParameterList_,"ExtensionSolver"),
                                                                 string("ExtensionSolver (Level ") + to_string(this->LevelID_) + string(")"));
                            this->ExtensionSolver_->initialize();
                            this->ExtensionSolver_->compute();
                            this->ExtensionSolver_->apply( *k_Re__MV, *inv_k_RR__k_Re__MV );  // (*input,*solution)
                            FROSCH_TIMER_STOP(timeFacesAGDSW7_loop_4);

                            FROSCH_TIMER_START_LEVELID(timeFacesAGDSW7_loop_5,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (7): loop [5]");
                            XMultiVectorPtr k_eR__inv_k_RR__k_Re__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(k_eR->getRowMap(),numFaceNodes);
                            k_eR->apply( *inv_k_RR__k_Re__MV, *k_eR__inv_k_RR__k_Re__MV );  // (*input,*solution)
		    
                            XMultiVectorPtr k_ee__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(k_ee->getRowMap(),numFaceNodes);
                            k_ee->apply( *id_e, *k_ee__MV );  // (*input,*solution)
	    
                            // This matrix will later store the local Schur complement.
                            // This Schur complement is the sum of subdomain-local Schur complements, e.g., S_ee__ij = S_ee_i + S_ee_j for an edge/face in two dimensions.
                            XMultiVectorPtr s_ee__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(k_ee->getRowMap(),numFaceNodes);
                            // this = gamma*this + alpha*A + beta*B
                            // update (alpha, A, beta, B, gamma)
                            s_ee__MV->update(ScalarTraits<SC>::one(),*k_ee__MV,  -ScalarTraits<SC>::one(),*k_eR__inv_k_RR__k_Re__MV,  ScalarTraits<SC>::zero());
                            // s_ee__MV should be the Schur complement with respect to one subdomain.
                            // It will subsequently be added with the Schur complement from the other subdomain s.t. we obtain the matrix for the eigenvalue problem of the adaptive coarse space.
                            evpLHSs.push_back(s_ee__MV);
                            evpRHSs.push_back(k_ee__MV);
                            FROSCH_TIMER_STOP(timeFacesAGDSW7_loop_5);
                        } // for: iterate over local faces
                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();
                        FROSCH_TIMER_STOP(timeFacesAGDSW7_loop);

                        // In the following four loops, the local Schur complements (from all subdomains 
                        // adjacent to the item (e.g., the edge)) are added to obtain the item Schur 
                        // complement. First, the local Schur complements are written to a repeated 
                        // MultiVector. Then, an exporter sums the values on the interface to obtain
                        // a uniquely distributed MultiVector. This is then distributed back to a 
                        // repeated MultiVector using an importer.

                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW8_loop,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (8): sum over local schur complements");

                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW81_loop,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (8): [1] beginExport");
                        // Write subdomain Schur complements into index-repeated matrix. Later, these (differing) entries will be summed over.
                        for (int localFaceID = 0; localFaceID < numFacesLocal; localFaceID++) {

                            // TODO: Kann es sein, dass localEntityID und localFaceID gleich sind? Ist das gesichert? Eines basiert auf globaler Sortierung. Die lokale koennte anders sein.
                            LO localEntityID = localEntityIDsOfSubdomain.at(localFaceID);
                            const InterfaceEntityPtr entity_ptr = DDInterface_->getFaces()->getEntity(localEntityID);
                            int numFaceNodes = entity_ptr->getNumNodes();

                            // Copy data from local MultiVector to global distributed MultiVector.
                            // The global distributed MultiVector is repeated and the data each process holds is the same data that the local MultiVector holds.
                            // --> This is a completely local operation.
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> s_ee__MV__repeated = s_ee__MV__repeated__list.at(localFaceID);
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> k_ee__MV__repeated = k_ee__MV__repeated__list.at(localFaceID);
                            XMultiVectorPtr s_ee__MV = evpLHSs.at(localFaceID);
                            XMultiVectorPtr k_ee__MV = evpRHSs.at(localFaceID);
                            for (int index_col = 0; index_col < numFaceNodes; index_col++) {
                                for (int index_row = 0; index_row < numFaceNodes; index_row++) {
                                    const SC val_s = s_ee__MV->getData(index_col)[index_row];
                                    s_ee__MV__repeated->getDataNonConst(index_col)[index_row] = val_s;

                                    const SC val_k = k_ee__MV->getData(index_col)[index_row];
                                    k_ee__MV__repeated->getDataNonConst(index_col)[index_row] = val_k;
                                }
                            }
            			}
                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();
                        FROSCH_TIMER_STOP(timeFacesAGDSW81_loop);

                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW81b_loop,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (8): [1b] beginExport");
                        Teuchos::Array< Teuchos::RCP<Tpetra::Export<LO, GO, NO>> > exporter__list(0);
                        // beginExport loop: from repeated to unique (add local Schur complements)
                        for (int localFaceID = 0; localFaceID < numFacesLocal; localFaceID++) {
                            // Create exporter to copy the data from the global repeated MultiVector to the global unique MultiVector.
                            Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> globalRepeatedMapForItem = itemMapsRepeated.at(localFaceID);
                            Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> globalUniqueMapForItem = itemMapsUnique.at(localFaceID);
                            Teuchos::RCP<Tpetra::Export<LO, GO, NO>> exporter = Teuchos::rcp(new Tpetra::Export<LO, GO, NO>(globalRepeatedMapForItem, globalUniqueMapForItem));
                            exporter__list.push_back(exporter);

                            // Export the data (sum over repeated (i.e., all) indices).
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> s_ee__MV__repeated = s_ee__MV__repeated__list.at(localFaceID);
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> k_ee__MV__repeated = k_ee__MV__repeated__list.at(localFaceID);
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> s_ee__MV__unique = s_ee__MV__unique__list.at(localFaceID);
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> k_ee__MV__unique = k_ee__MV__unique__list.at(localFaceID);
                            s_ee__MV__unique->beginExport(*s_ee__MV__repeated, *exporter, Tpetra::ADD);
                            k_ee__MV__unique->beginExport(*k_ee__MV__repeated, *exporter, Tpetra::ADD);
                        }
                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();
                        FROSCH_TIMER_STOP(timeFacesAGDSW81b_loop);

                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW82_loop,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (8): [2] endExport");
                        // endExport loop
                        for (int localFaceID = 0; localFaceID < numFacesLocal; localFaceID++) {
                            Teuchos::RCP<Tpetra::Export<LO, GO, NO>> exporter = exporter__list.at(localFaceID);

                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> s_ee__MV__unique   = s_ee__MV__unique__list.at(localFaceID);
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> k_ee__MV__unique   = k_ee__MV__unique__list.at(localFaceID);
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> s_ee__MV__repeated = s_ee__MV__repeated__list.at(localFaceID);
                            Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> k_ee__MV__repeated = k_ee__MV__repeated__list.at(localFaceID);

                            s_ee__MV__unique->endExport(*s_ee__MV__repeated, *exporter, Tpetra::ADD);
                            k_ee__MV__unique->endExport(*k_ee__MV__repeated, *exporter, Tpetra::ADD);
                        }
                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();
                        FROSCH_TIMER_STOP(timeFacesAGDSW82_loop);

                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW83_loop,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (8): [3] create Importer");
                        Teuchos::Array< Teuchos::RCP<Tpetra::Import<LO, GO, NO>> > importer__list(0);
                        // beginImport loop: from unique to repeated (distribute Schur complement of interface component)
                        for (int localFaceID = 0; localFaceID < numFacesLocal; localFaceID++) {
                            // Create importer to copy data from global unique MultiVector to global repeated MultiVector.
                            Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> globalRepeatedMapForItem = itemMapsRepeated.at(localFaceID);
                            Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> globalUniqueMapForItem = itemMapsUnique.at(localFaceID);
                            Teuchos::RCP<Tpetra::Import<LO, GO, NO>> importer = Teuchos::rcp(new Tpetra::Import<LO, GO, NO>(globalUniqueMapForItem,globalRepeatedMapForItem));
                            importer__list.push_back(importer);
                        }
                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();
                        FROSCH_TIMER_STOP(timeFacesAGDSW83_loop);

                        FROSCH_TIMER_STOP(timeFacesAGDSW8_loop);

                        // Solve eigenvalue problems.
                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW9_loop,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (9): solve all local EVPs");
                        Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > > schur_ptr;
                        Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > > matrixB_ptr;
        				Teuchos::Array<Teuchos::RCP<GOVec> > allLocalToGlobalVectors(numFaces_global);
                        std::vector<int> numEigVec__list(0);
                        int sumNumFn = 0;               // total number of functions on current rank (managed or not)
                        int sumNumFnHandledByRoot = 0;  // number of functions managed by current rank
                        Teuchos::Array< Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> > > eigenvectors_ptr__list(0);
                        Teuchos::Array< LOVec > selectedEigenvectors__list(0);
                        for (int localFaceID = 0; localFaceID < numFacesLocal; localFaceID++) {
                            Teuchos::RCP< const Teuchos::Comm<int> > commNeighborsOfEntity = subcomms.at(localFaceID);
                            GO globalFaceID = globalFaceIDsOfSubdomain.at(localFaceID);

                            // TODO: Kann es sein, dass localEntityID und localFaceID gleich sind? Ist das gesichert? Eines basiert auf globaler Sortierung. Die lokale koennte anders sein.
                            LO localEntityID = localEntityIDsOfSubdomain.at(localFaceID);
                            const InterfaceEntityPtr entity_ptr = DDInterface_->getFaces()->getEntity(localEntityID);
                            int numFaceNodes = entity_ptr->getNumNodes();

                            Teuchos::RCP< std::vector<SC> > eigenvalues_ptr;
                            Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> > eigenvectors_ptr;
                            LOVec sel(0);
                            int numEigVecToSelect = 0;

                            // Only one of the ranks shall solve the eigenvalue problem.
                            // For one thing, it is not necessary for all to do the job (even if it may be cheap).
                            // For another thing, rounding errors may affect the solution ever so slightly and lead to differently 
                            // ordered eigenfunctions on the associated MPI ranks. This will break the coarse space.
                            // Lastly, this simplifies the storage of eigenfunctions later on.
                            if (commNeighborsOfEntity->getRank() == rootRankOfNeighborhood) {
                                FROSCH_TIMER_START_LEVELID(timeFacesAGDSW9_loop_1,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (9): loop [1]");

                                // TODO: Re-add the export of the Schur complements as an option.
                                // Export edge Schur complements.
                                // Xpetra::IO< SC, LO, GO, NO >::Write("s_ee__ij__e="+std::to_string(ii)+".txt", *s_ee__ij, true);
                                // Teuchos::RCP< Xpetra::Map<LO,GO,NO> > faceNodeMap = MapFactory<LO,GO,NO>::Build(this->K_->getRowMap()->lib(),INVALID,itemNodesGlobalRepeated(),0,this->MpiComm_);
                                // This does not extract the subdomain matrices but the matrices corresponding to the entity nodes.
                                // ConstXMatrixPtr repeatedMatrixS__ = FROSch::ExtractLocalSubdomainMatrix(s_ee__ij.getConst(),faceNodeMap.getConst());
                                // ConstXMatrixPtr repeatedMatrixKee__ = FROSch::ExtractLocalSubdomainMatrix(k_ee__ij.getConst(),faceNodeMap.getConst());

                                Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> s_ee__MV__unique = s_ee__MV__unique__list.at(localFaceID);
                                Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> k_ee__MV__unique = k_ee__MV__unique__list.at(localFaceID);

                                schur_ptr = FROSch::convert_GlobalTMultiVector_to_SerialDenseMatrix(s_ee__MV__unique.getConst());
                                matrixB_ptr = FROSch::convert_GlobalTMultiVector_to_SerialDenseMatrix(k_ee__MV__unique.getConst());
                                FROSCH_TIMER_STOP(timeFacesAGDSW9_loop_1);

                                FROSCH_TIMER_START_LEVELID(timeFacesAGDSW9_loop_2,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (9): loop [2] EVP Solving");
                                // Solve SchurComplement * x = lambda * B * x.
                                using Matrix_Dense_ptr = Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > >;
                                FROSch::EigenSolverFactory<Matrix_Dense_ptr , Matrix_Dense_ptr>::Solve(
                                    schur_ptr,
                                    matrixB_ptr,
                                    parameterList_adaptiveProblems,
                                    eigenvalues_ptr,
                                    eigenvectors_ptr);
                                FROSCH_TIMER_STOP(timeFacesAGDSW9_loop_2);

                                FROSCH_TIMER_START_LEVELID(timeFacesAGDSW9_loop_3,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (9): loop [3]");
                                const double tol = parameterList_adaptiveProblems->get("Tolerance for the selection of functions", 0.01);
                                for (LO kk = 0; kk < (LO)eigenvalues_ptr->size(); kk++) {
                                    if ((*eigenvalues_ptr)[kk] < tol) {
                                        numEigVecToSelect += 1;
                                        sel.push_back(kk);
                                    }
                                }

                                sumNumFnHandledByRoot += numEigVecToSelect;
                                if ((numEigVecToSelect == 0) && (parameterList_adaptiveProblems->get("Include GDSW functions", false))) {
                                    sumNumFnHandledByRoot += 1;
                                }

                                eigenvectors_ptr__list.push_back(eigenvectors_ptr);
                                selectedEigenvectors__list.push_back(sel);
                                FROSCH_TIMER_STOP(timeFacesAGDSW9_loop_3);
                            } else {
                                eigenvectors_ptr__list.push_back(Teuchos::null);
                                selectedEigenvectors__list.push_back(LOVec(0));
                            }
                            numEigVec__list.push_back(numEigVecToSelect);
                        }

                        // Broadcast selected number of eigenvectors to all affected subdomains / cores.
                        for (int localFaceID = 0; localFaceID < numFacesLocal; localFaceID++) {
                            Teuchos::RCP< const Teuchos::Comm<int> > commNeighborsOfEntity = subcomms.at(localFaceID);
                            int numEigVecToSelect = numEigVec__list.at(localFaceID);
                            Teuchos::broadcast<int, int>(*commNeighborsOfEntity, rootRankOfNeighborhood, 1, &numEigVecToSelect);
                            numEigVec__list.at(localFaceID) = numEigVecToSelect;

                            sumNumFn += numEigVecToSelect;
                            if ((numEigVecToSelect == 0) && (parameterList_adaptiveProblems->get("Include GDSW functions", false))) {
                                sumNumFn += 1;
                            }
                        }

                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();
                        FROSCH_TIMER_STOP(timeFacesAGDSW9_loop);

                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW10_loop,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (10): compute offsets");
                        // Since the selected eigenvectors overlap (are generally shared by multiple subdomains), 
                        // above, we have computed the sum of the numbers of only those selected eigenvectors that are held
                        // on rank 0 (within the neighborhood communicator).
                        // Now, we can compute the cumulative sum across all ranks (of the neighborhood communicator).
                        // This will give us an offset, in which column the eigenvectors shall be stored globally.
                        int cumsum = -1;
                        Teuchos::scan<int, int>(*(this->MpiComm_), Teuchos::REDUCE_SUM, sumNumFnHandledByRoot, Teuchos::outArg(cumsum));
                        const int offset = cumsum - sumNumFnHandledByRoot;

                        // Compute offset per item.
                        // For example, the first edge may be held by the current rank. The offset will be 
                        // the offset determined above + 0. We assume that the first edge has 4 selected eigenvectors.
                        // The second edge may also be held be the current rank. The offset will be "offset + 4".
                        // The third edge is held by a neighboring rank. This rank will have to tell the current rank
                        // what the offset should be.
                        int offset_counter = offset;
                        std::vector<int> offset_item__list(0);
                        for (int localFaceID = 0; localFaceID < numFacesLocal; localFaceID++) {
                            Teuchos::RCP< const Teuchos::Comm<int> > commNeighborsOfEntity = subcomms.at(localFaceID);
                            int offset_item = -1;
                            if (commNeighborsOfEntity->getRank() == rootRankOfNeighborhood) {
                                offset_item = offset_counter;

                                const int numEigVecToSelect = numEigVec__list.at(localFaceID);
                                offset_counter += numEigVecToSelect;
                                if ((numEigVecToSelect == 0) && (parameterList_adaptiveProblems->get("Include GDSW functions", false))) {
                                    offset_counter += 1;
                                }
                            }
                            Teuchos::broadcast<int, int>(*commNeighborsOfEntity, rootRankOfNeighborhood, 1, &offset_item);
                            offset_item__list.push_back(offset_item);
                        }
                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();
                        FROSCH_TIMER_STOP(timeFacesAGDSW10_loop);

                        FROSCH_TIMER_START_LEVELID(timeFacesAGDSW11_loop,"GDSWCoarseOperator::resetCoarseSpaceBlock::AGDSW face functions (11): store functions");
                        Teuchos::RCP<GOVec> localToGlobalVector = Teuchos::rcp(new GOVec(0));
                        if (sumNumFn > 0) {
                            translations[0] = MultiVectorFactory<SC,LO,GO,NO>::Build(serialGammaMap, sumNumFn);
                        } else {
                            translations[0] = Teuchos::null;
                        }

                        int localOffset = 0;
                        for (int localFaceID = 0; localFaceID < numFacesLocal; localFaceID++) {

                            int numEigVecToSelect = numEigVec__list[localFaceID];
                            Teuchos::RCP< const Teuchos::Comm<int> > commNeighborsOfEntity = subcomms.at(localFaceID);
                            Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> > eigenvectors_ptr = eigenvectors_ptr__list.at(localFaceID);

                            LO localEntityID = localEntityIDsOfSubdomain.at(localFaceID);
                            const InterfaceEntityPtr entity_ptr = DDInterface_->getFaces()->getEntity(localEntityID);
                            int numFaceNodes = entity_ptr->getNumNodes();

                            // Each MPI rank needs to have access to the eigenvectors.
                            // In the following, we will extract the selected eigenvectors, store them in a uniquely-distributed MultiVector, 
                            // copy this over to a repeatedly-distributed MultiVector (s.t. all item-associated MPI ranks have access to the values), 
                            // and then copy the values to a SerialDenseMatrix. TODO: not necessary.
                            Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > > selectedEigenvectors_ptr;
                            if (numEigVecToSelect > 0) {
                                Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> globalUniqueMapForItem = itemMapsUnique.at(localFaceID);
                                Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> globalRepeatedMapForItem = itemMapsRepeated.at(localFaceID);
                                Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> selectedEigenvectors__MV__unique = Teuchos::rcp(new Tpetra::MultiVector<SC, LO, GO, NO>(globalUniqueMapForItem, numEigVecToSelect));
                                Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> selectedEigenvectors__MV__repeated = Teuchos::rcp(new Tpetra::MultiVector<SC, LO, GO, NO>(globalRepeatedMapForItem, numEigVecToSelect));

                                if (commNeighborsOfEntity->getRank() == rootRankOfNeighborhood) {

                                    LOVec sel = selectedEigenvectors__list.at(localFaceID);

                                    // std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
                                    // Teuchos::RCP< const Teuchos::Comm<int> > commNeighborsOfEntity = subcomms.at(localFaceID);
                                    // commNeighborsOfEntity->barrier();
                                    // std::cout << "Eigenvalues: " << std::endl;
                                    // for (LO kk = 0; kk < (LO)eigenvalues_ptr->size(); kk++) {
                                    //     std::cout << "i = " << kk << ": " << (*eigenvalues_ptr)[kk] << std::endl;
                                    // }
                                    // std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
                                    // commNeighborsOfEntity->barrier();
    
                                    // Copy eigenvectors from SerialDenseMatrix to MultiVector that is shared by all item-associated MPI ranks.
                                    // All values (of the matrix) are still stored on the current rank, since the unique map was set up this way.
                                    // In a next step, the MultiVector will be distributed to the other item-associated MPI ranks, 
                                    // using a "unique-map to repeated-map" distribution (Tpetra Import).
                                    for (int index_eigenvec = 0; index_eigenvec < numEigVecToSelect; index_eigenvec++) {
                                        int index_col = sel[index_eigenvec];
                                        for (int index_row = 0; index_row < numFaceNodes; index_row++) {
                                            const SC val = (*eigenvectors_ptr)(index_row, index_col);
                                            selectedEigenvectors__MV__unique->getDataNonConst(index_eigenvec)[index_row] = val;
                                        }
                                    }
                                }

                                // Import the data (distribute, i.e., duplicate data from unique indices to repeated indices)
                                Teuchos::RCP<Tpetra::Import<LO, GO, NO>> importer = importer__list.at(localFaceID);
                                selectedEigenvectors__MV__repeated->doImport(*selectedEigenvectors__MV__unique, *importer, Tpetra::INSERT); // TODO: beginImport/endImport
    
                                selectedEigenvectors_ptr = FROSch::convert_GlobalTMultiVector_to_SerialDenseMatrix(selectedEigenvectors__MV__repeated.getConst());
                            }

                            if (numEigVecToSelect > 0) {
                                for (int eigfn = 0; eigfn < numEigVecToSelect; eigfn++) {
                                    for (int j = 0; j < numFaceNodes; j++) {
                                        translations[0]->replaceLocalValue( entity_ptr->getGammaDofID(j,0), localOffset + eigfn, (*selectedEigenvectors_ptr)(j,eigfn) );
                                    }
                                    localToGlobalVector->push_back(offset_item__list[localFaceID] + eigfn);
                                }
                                localOffset += numEigVecToSelect;
                            } else {
                                if (parameterList_adaptiveProblems->get("Include GDSW functions", false)) {
                                    for (int j = 0; j < numFaceNodes; j++) {
                                        translations[0]->replaceLocalValue( entity_ptr->getGammaDofID(j,0), localOffset, ScalarTraits<SC>::one() );
                                    }
                                    localToGlobalVector->push_back(offset_item__list[localFaceID]);
                                    localOffset += 1;
                                }
                            }
                        } // for: iterate over local faces

                        // Store interface coarse space.
                        const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid();
                        ConstXMapPtr facesEntityMap = MapFactory<LO,GO,NO>::Build(this->K_->getRowMap()->lib(),INVALID,*localToGlobalVector(),0,this->MpiComm_);
                        //std::cout << this->MpiComm_->getRank() << "  " << translations[0]->getNumVectors() << "  " << facesEntityMap->getLocalNumElements() << std::endl;
                        this->InterfaceCoarseSpaces_[blockId]->addSubspace(facesEntityMap,null,translations[0]);

                        if (addMPIBarriersForSomeTimers) this->MpiComm_->barrier();
                        FROSCH_TIMER_STOP(timeFacesAGDSW11_loop);

                        FROSCH_TIMER_STOP(timeFacesAGDSW);
                    }

                    if (useFaceRotations) {
                        XMultiVectorPtrVecPtr rotations = this->computeRotations(blockId,dimension,nodeList,DDInterface_->getFaces());
                        ConstXMapPtr facesEntityMap = DDInterface_->getFaces()->getEntityMap();
                        for (UN i=0; i<rotations.size(); i++) {
                            this->InterfaceCoarseSpaces_[blockId]->addSubspace(facesEntityMap,null,rotations[i]);
                        }
                    }

                    this->InterfaceCoarseSpaces_[blockId]->assembleCoarseSpace();
//                    this->InterfaceCoarseSpaces_[blockId]->assembleCoarseSpaceT();

                    if (this->Verbose_) {
                        cout
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << setw(89) << "-----------------------------------------------------------------------------------------"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << "| "
                        << left << setw(74) << "> GDSW coarse space " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")"
                        << " |"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << setw(89) << "========================================================================================="
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << "| " << left << setw(19) << "Vertices " << " | " << setw(19) << "Translations " << right
                        << " | " << setw(41) << boolalpha << useVertexTranslations << noboolalpha
                        << " |"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << "| " << left << setw(19) << "ShortEdges " << " | " << setw(19) << "Translations " << right
                        << " | " << setw(41) << boolalpha << useShortEdgeTranslations << noboolalpha
                        << " |"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << "| " << left << setw(19) << "ShortEdges " << " | " << setw(19) << "Rotations " << right
                        << " | " << setw(41) << boolalpha << useShortEdgeRotations << noboolalpha
                        << " |"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << "| " << left << setw(19) << "StraightEdges " << " | " << setw(19) << "Translations " << right
                        << " | " << setw(41) << boolalpha << useStraightEdgeTranslations << noboolalpha
                        << " |"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << "| " << left << setw(19) << "StraightEdges " << " | " << setw(19) << "Rotations " << right
                        << " | " << setw(41) << boolalpha << useStraightEdgeRotations << noboolalpha
                        << " |"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << "| " << left << setw(19) << "Edges " << " | " << setw(19) << "Translations " << right
                        << " | " << setw(41) << boolalpha << useEdgeTranslations << noboolalpha
                        << " |"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << "| " << left << setw(19) << "Edges " << " | " << setw(19) << "Rotations " << right
                        << " | " << setw(41) << boolalpha << useEdgeRotations << noboolalpha
                        << " |"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << "| " << left << setw(19) << "Faces " << " | " << setw(19) << "Translations " << right
                        << " | " << setw(41) << boolalpha << useFaceTranslations << noboolalpha
                        << " |"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << "| " << left << setw(19) << "Faces " << " | " << setw(19) << "Rotations " << right
                        << " | " << setw(41) << boolalpha << useFaceRotations << noboolalpha
                        << " |"
                        << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                        << setw(89) << "-----------------------------------------------------------------------------------------"
                        << endl;
                    }
                }
            }
        }
        return 0;
    }
}

#endif
