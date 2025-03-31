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

#include <FROSch_GDSWCoarseOperator_decl.hpp>

#include <FROSch_EigenSolver_def.hpp>
#include <FROSch_EigenSolverFactory_def.hpp>
#include "Tpetra_FECrsMatrix.hpp"

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

                    // Faces
                    if (useFaceTranslations) {
                        bool useAdaptiveCoarseSpace = this->ParameterList_->get("Use Adaptive Coarse Space",true);

                        if (!useAdaptiveCoarseSpace) {
                            XMultiVectorPtrVecPtr translations = this->computeTranslations(blockId,DDInterface_->getFaces());
                            ConstXMapPtr facesEntityMap = DDInterface_->getFaces()->getEntityMap();

                            FROSch::debug::printMap(facesEntityMap, "facesEntityMap", __FILE__, __LINE__);

                            for (UN i=0; i<translations.size(); i++) {
                                this->InterfaceCoarseSpaces_[blockId]->addSubspace(facesEntityMap,null,translations[i]);
                            }
                        } else {
                            XMultiVectorPtrVecPtr translations = XMultiVectorPtrVecPtr(1);

                            this->ParameterList_->print();

                            Teuchos::RCP< Tpetra::FECrsMatrix<SC,LO,GO,NO> > dummy = Teuchos::null;
                            Teuchos::RCP< Tpetra::FECrsMatrix<SC,LO,GO,NO> > fe_matrix = this->ParameterList_->get("Neumann Matrices",dummy);

                            ConstXMapPtr repeatedMap;
                            ConstXMatrixPtr repeatedMatrix;
                            repeatedMap = FROSch::AssembleSubdomainMap(this->NumberOfBlocks_,this->DofsMaps_,this->DofsPerNode_);
                            if (fe_matrix == Teuchos::null) {
                                repeatedMatrix = FROSch::ExtractLocalSubdomainMatrix(this->K_,repeatedMap.getConst());
                            } else {
                                repeatedMatrix = FROSch::ExtractLocalSubdomainMatrix_feTest(fe_matrix.getConst(),repeatedMap.getConst());
                            }

                            // Extract submatrices
                            const int numEl = nodesMap->getLocalNumElements();
                            std::vector<GO> allLocalNodes(numEl); // not dofs! needs to be changed.
                            for (int ii = 0; ii < numEl; ii++) {
                                allLocalNodes[ii] = ii;
                            }

                            const int numFaces_global = (int)DDInterface_->getFaces()->getEntityMap()->getMaxAllGlobalIndex()+1; // +1 since indices are zero based

                            XMapPtr serialGammaMap = MapFactory<LO,GO,NO>::Build(this->K_->getRowMap()->lib(),this->GammaDofs_[0].size(),0,this->SerialComm_);
                            //const int numFaces_local = DDInterface_->getFaces()->getNumEntities();

                            for (int ii = 0; ii < numFaces_global; ii++){
                                const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid();
                                const LO localEntityID = DDInterface_->getFaces()->getEntityMap()->getLocalElement(ii);

                                // Create split communicator to those subdomains neighboring the entity and the remaining ones.
                                const bool isSubdomainNeighborOfEntity = (localEntityID != INVALID);

                                int color;
                                if (isSubdomainNeighborOfEntity) {
                                    color = 0;
                                } else {
                                    color = 1;
                                }
                                Teuchos::RCP< const Teuchos::Comm<int> > commNeighborsOfEntity = 
                                    this->MpiComm_->split(color, this->MpiComm_->getRank());

                                // get number of face nodes
                                int numFaceNodes = 0;
                                using InterfaceEntityPtr = typename SchwarzOperator<SC,LO,GO,NO>::InterfaceEntityPtr;
                                if (localEntityID != INVALID) {
                                    const InterfaceEntityPtr entity_ptr = DDInterface_->getFaces()->getEntity(localEntityID);
                                    numFaceNodes = entity_ptr->getNumNodes();
                                }
                                int maxNumFaceNodes_ranks = 0;
                                reduceAll(*this->MpiComm_,Teuchos::REDUCE_MAX,numFaceNodes,ptr(&maxNumFaceNodes_ranks));
                                numFaceNodes = maxNumFaceNodes_ranks;

                                // This matrix will later store the local Schur complement.
                                // This Schur complement is the sum of subdomain-local Schur complements, e.g., S_ee__ij = S_ee_i + S_ee_j for an edge/face in two dimensions.
                                XMatrixPtr s_ee__ij = Xpetra::MatrixFactory<SC,LO,GO,NO>::Build(this->K_->getRowMap(),numFaceNodes);
                                XMatrixPtr k_ee__ij = Xpetra::MatrixFactory<SC,LO,GO,NO>::Build(this->K_->getRowMap(),numFaceNodes);
    			    
                                GOVec indicesR(0);  // R:[r]emaining nodes
    			    
                                std::vector<GO> itemNodes(numFaceNodes); // not dofs! needs to be changed.
                                Teuchos::Array<GO> itemNodes__A(0);
                                XMatrixPtr k_ee;
                                InterfaceEntityPtr entity_ptr;
                                if (localEntityID != INVALID) {
                                    entity_ptr = DDInterface_->getFaces()->getEntity(localEntityID);
    			    
                                    // Get entity nodes.
                                    for (int jj = 0; jj < numFaceNodes; jj++) {
                                        itemNodes[jj] = entity_ptr->getNode(jj).NodeIDLocal_;
                                        itemNodes__A.push_back(entity_ptr->getNode(jj).NodeIDGlobal_);
                                    }
    			    
                                    // Get set of remaining subdomain nodes.
                                    std::vector<int> diff;
                                    std::set_difference(allLocalNodes.begin(), allLocalNodes.end(), itemNodes.begin(), itemNodes.end(), std::inserter(diff, diff.begin()));
                                    for (int jj = 0; jj < (int)diff.size(); jj++) {
                                        indicesR.push_back(diff[jj]);
                                    }
    			    
                                    XMatrixPtr k_RR;
                                    XMatrixPtr k_Re;
                                    XMatrixPtr k_eR;
    			    
                                    FROSch::BuildSubmatrices(repeatedMatrix.getConst(),indicesR(),k_RR,k_Re,k_eR,k_ee);
			    
                                    // [JK] Todo: assert: numFaceNodes == k_ee->getRowMap()->getLocalNumElements()
                                    XMultiVectorPtr id_e = MultiVectorFactory<SC,LO,GO,NO>::Build(k_ee->getRowMap(),numFaceNodes);
                                    for (int jj=0; jj<numFaceNodes; jj++) {
                                        id_e->replaceLocalValue(jj,jj,ScalarTraits<SC>::one());
                                    }
        		    
                                    XMultiVectorPtr k_Re__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(k_Re->getRowMap(),numFaceNodes);
                                    k_Re->apply( *id_e, *k_Re__MV );  // (*input,*solution)
        		    
                                    // Solve k_RR * X = k_Re__MV.
                                    // --> inv_k_RR__k_Re__MV := X.
                                    XMultiVectorPtr inv_k_RR__k_Re__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(k_RR->getRowMap(),numFaceNodes);
                                    this->ExtensionSolver_ = SolverFactory<SC,LO,GO,NO>::Build(k_RR,
                                                                         sublist(this->ParameterList_,"ExtensionSolver"),
                                                                         string("ExtensionSolver (Level ") + to_string(this->LevelID_) + string(")"));
                                    this->ExtensionSolver_->initialize();
                                    this->ExtensionSolver_->compute();
                                    this->ExtensionSolver_->apply( *k_Re__MV, *inv_k_RR__k_Re__MV );  // (*input,*solution)
        		    
                                    XMultiVectorPtr k_eR__inv_k_RR__k_Re__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(k_eR->getRowMap(),numFaceNodes);
                                    k_eR->apply( *inv_k_RR__k_Re__MV, *k_eR__inv_k_RR__k_Re__MV );  // (*input,*solution)
        		    
                                    XMultiVectorPtr k_ee__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(k_ee->getRowMap(),numFaceNodes);
                                    k_ee->apply( *id_e, *k_ee__MV );  // (*input,*solution)
			    
                                    XMultiVectorPtr s_ee__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(k_ee->getRowMap(),numFaceNodes);
                                    // this = gamma*this + alpha*A + beta*B
                                    // update (alpha, A, beta, B, gamma)
                                    s_ee__MV->update(ScalarTraits<SC>::one(),*k_ee__MV,  -ScalarTraits<SC>::one(),*k_eR__inv_k_RR__k_Re__MV,  ScalarTraits<SC>::zero());
                                    // s_ee__MV should be the Schur complement with respect to one subdomain.
                                    // It will subsequently be added with the Schur complement from the other subdomain s.t. we obtain the matrix for the eigenvalue problem of the adaptive coarse space.
        		    
                                    // Write Schur complement into global matrix, extract from global Stiffness matrix the values correponding to the interface item.
                                    for (int jj = 0; jj < numFaceNodes; jj++) {
                                        const Array<GO> globalID_jj(1,entity_ptr->getNode(jj).NodeIDGlobal_);
                                        for (int kk = 0; kk < numFaceNodes; kk++) {
                                            const Array<GO> globalID_kk(1,entity_ptr->getNode(kk).NodeIDGlobal_);
                                            const SC val = s_ee__MV->getData(jj)[kk];
                                            const Array<SC> vall(1,val);
                                            s_ee__ij->insertGlobalValues(globalID_jj[0],globalID_kk(),vall());
        		    
                                            const SC val_ = k_ee__MV->getData(jj)[kk];
                                            const Array<SC> vall_(1,val_);
                                            k_ee__ij->insertGlobalValues(globalID_jj[0],globalID_kk(),vall_());
                                        }
                                    }
                                }
        		    
                                s_ee__ij->fillComplete(this->K_->getMap(),this->K_->getMap());
                                k_ee__ij->fillComplete(this->K_->getMap(),this->K_->getMap());
        		    
                                // Export edge Schur complements.
                                //Xpetra::IO< SC, LO, GO, NO >::Write("s_ee__ij__e="+std::to_string(ii)+".txt", *s_ee__ij, true);
        		    
                                Teuchos::RCP< Xpetra::Map<LO,GO,NO> > faceNodeMap = MapFactory<LO,GO,NO>::Build(this->K_->getRowMap()->lib(),INVALID,itemNodes__A(),0,this->MpiComm_);
        		    
                                Array<GO> itemNodes__(0);
                                if (localEntityID != INVALID) {      
                                    // For each subdomain that contains the interface component:
                                    // Extract the Schur complement (corresponding to the interface component) from the global sparse matrix.
                                    for (int jj = 0; jj < numFaceNodes; jj++) {
                                        itemNodes__.push_back(itemNodes[jj]);
                                    }
                                }
        		    
                                // This does not extract the subdomain matrices but the matrices corresponding to the entity nodes.
                                ConstXMatrixPtr repeatedMatrixS__ = FROSch::ExtractLocalSubdomainMatrix(s_ee__ij.getConst(),faceNodeMap.getConst());
                                ConstXMatrixPtr repeatedMatrixKee__ = FROSch::ExtractLocalSubdomainMatrix(k_ee__ij.getConst(),faceNodeMap.getConst());

                                Teuchos::RCP< std::vector<SC> > eigenvalues_ptr;
                                Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> > eigenvectors_ptr;
                                LOVec sel;
                                int numEigVecToSelect = 0;

                                if (localEntityID != INVALID) {
                                    // assert: numFaceNodes = size(s_ee__ij_local)
                                    //XMultiVectorPtr k_Re__MV = MultiVectorFactory<SC,LO,GO,NO>::Build(s_ee__ij_local->getRowMap(),numFaceNodes);
                                    //k_Re->apply(*id_e,*k_Re__MV);
                                    Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > > schur_ptr = FROSch::convert_LocalSquareXMatrix_to_SerialDenseMatrix(repeatedMatrixS__);
                                    //Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > > matrixB_ptr = FROSch::convert_LocalSquareXMatrix_to_SerialDenseMatrix(k_ee.getConst());
                                    Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > > matrixB_ptr = FROSch::convert_LocalSquareXMatrix_to_SerialDenseMatrix(repeatedMatrixKee__);

                                    Teuchos::RCP<Teuchos::ParameterList> parameterList_adaptiveProblems = Teuchos::sublist(this->ParameterList_, "Adaptive problems");

                                    // TODO: It should not be necessary to create an object for the eigensolver.
                                    using Matrix_Dense_ptr = Teuchos::RCP< Teuchos::SerialDenseMatrix< LO, SC > >;
                                    FROSch::EigenSolverFactory<Matrix_Dense_ptr , Matrix_Dense_ptr>::Build(
                                        schur_ptr,
                                        matrixB_ptr,
                                        parameterList_adaptiveProblems,
                                        eigenvalues_ptr,
                                        eigenvectors_ptr);
                                    
                                    const double tol = parameterList_adaptiveProblems->get("Tolerance for the selection of functions", 0.01);
                                    for (LO kk = 0; kk < (LO)eigenvalues_ptr->size(); kk++) {
                                        if ((*eigenvalues_ptr)[kk] < tol) {
                                            numEigVecToSelect += 1;
                                            sel.push_back(kk);
                                        }
                                    }

/*
                                    std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
                                    commNeighborsOfEntity->barrier();
                                    std::cout << "Eigenvalues: " << std::endl;
                                    for (LO kk = 0; kk < (LO)eigenvalues_ptr->size(); kk++) {
                                        std::cout << "i = " << kk << ": " << (*eigenvalues_ptr)[kk] << std::endl;
                                    }
                                    std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
                                    commNeighborsOfEntity->barrier();
*/
                                }

                                GOVec localToGlobalVector(0);
                                if (numEigVecToSelect > 0) {
                                    translations[0] = MultiVectorFactory<SC,LO,GO,NO>::Build(serialGammaMap,numEigVecToSelect);
                                    for (int eigfn = 0; eigfn < numEigVecToSelect; eigfn++) {
                                        for (int j = 0; j < numFaceNodes; j++) {
                                            translations[0]->replaceLocalValue( entity_ptr->getGammaDofID(j,0), eigfn, (*eigenvectors_ptr)(j,sel[eigfn]) );
                                        }
                                        localToGlobalVector.push_back(eigfn);
                                    }
                                } else {
                                    translations[0] = Teuchos::null;
                                }

                                ConstXMapPtr facesEntityMap = MapFactory<LO,GO,NO>::Build(this->K_->getRowMap()->lib(),INVALID,localToGlobalVector(),0,this->MpiComm_);
                                this->InterfaceCoarseSpaces_[blockId]->addSubspace(facesEntityMap,null,translations[0]);
                            } // for: iterate over global faces
                        }
                    }

                    if (useFaceRotations) {
                        XMultiVectorPtrVecPtr rotations = this->computeRotations(blockId,dimension,nodeList,DDInterface_->getFaces());
                        ConstXMapPtr facesEntityMap = DDInterface_->getFaces()->getEntityMap();
                        for (UN i=0; i<rotations.size(); i++) {
                            this->InterfaceCoarseSpaces_[blockId]->addSubspace(facesEntityMap,null,rotations[i]);
                        }
                    }

                    this->InterfaceCoarseSpaces_[blockId]->assembleCoarseSpace();

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
