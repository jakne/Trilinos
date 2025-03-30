// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef FROSCHEXAMPLES_ADAPTIVE_GDSW_MESHDATABASE_HPP
#define FROSCHEXAMPLES_ADAPTIVE_GDSW_MESHDATABASE_HPP

// This file was taken and adapted from the Tpetra FECrs example "Finite-Element-Assembly".

#include <fstream>
#include <iostream>
#include <set>
#include <string>

#include <Teuchos_Comm.hpp>
#include <Tpetra_Details_WrappedDualView.hpp>

template <typename GO> struct LLA {
    LLA() {
        data[0] = 0;
        data[1] = 0;
    }

    LLA(GO i, GO j) {
        data[0] = i;
        data[1] = j;
    }

    GO operator[](GO i) const { return data[i]; }
    GO &operator[](GO i) { return data[i]; }

    GO data[2];
};

template <typename GO, typename NO> class MeshDatabase {
  public:
    using hostType = typename Kokkos::DualView<int *, typename NO::device_type>::t_host::device_type;
    using globalDualViewType = Kokkos::DualView<GO *, typename NO::device_type>;
    using global2DArrayDualViewType = Kokkos::DualView<GO *[4], typename NO::device_type>;
    using global_ordinal_view_type = Tpetra::Details::WrappedDualView<globalDualViewType>;
    using global_ordinal_2d_array_type = Tpetra::Details::WrappedDualView<global2DArrayDualViewType>;

    MeshDatabase(Teuchos::RCP<const Teuchos::Comm<int>> comm, GO num_global_elements_x, GO num_global_elements_y, GO num_procs_x, GO num_procs_y);

    ~MeshDatabase() {}

    // Size accessors
    size_t getNumOwnedElements() const { return ownedElementGlobalIDs_.extent(0); }

    size_t getNumGhostElements() const { return ghostElementGlobalIDs_.extent(0); }

    size_t getNumOwnedNodes() const { return ownedNodeGlobalIDs_.extent(0); }

    size_t getNumGhostNodes() const { return ghostNodeGlobalIDs_.extent(0); }

    size_t getNumOwnedAndGhostNodes() const { return ownedAndGhostNodeGlobalIDs_.extent(0); }

    size_t getNumOwnedAndGhostElements() const { return ownedAndGhostElementGlobalIDs_.extent(0); }

    // Data accessors
    global_ordinal_view_type getOwnedElementGlobalIDs() { return ownedElementGlobalIDs_; }
    global_ordinal_view_type getGhostElementGlobalIDs() { return ghostElementGlobalIDs_; }
    global_ordinal_view_type getOwnedAndGhostElementGlobalIDs() { return ownedAndGhostElementGlobalIDs_; }

    global_ordinal_view_type getOwnedNodeGlobalIDs() { return ownedNodeGlobalIDs_; }
    global_ordinal_view_type getGhostNodeGlobalIDs() { return ghostNodeGlobalIDs_; }
    global_ordinal_view_type getOwnedAndGhostNodeGlobalIDs() { return ownedAndGhostNodeGlobalIDs_; }

    global_ordinal_2d_array_type getOwnedElementToNode() { return ownedElementToNode_; }
    global_ordinal_2d_array_type getGhostElementToNode() { return ghostElementToNode_; }

    GO getNumGlobalElements() const { return numGlobalElements_[0] * numGlobalElements_[1]; }

    // Debugging output
    void print(std::ostream &oss);

    void exportToFiles(std::string base_file_name) const;

    inline bool nodeIsOwned(GO linear_index) const {
        GO row, col;
        rowCol_from_linear_index(numGlobalNodes_[0], linear_index, row, col);
        return nodeIsOwned(row, col);
    }

    inline bool elementIsOwned(GO linear_index) const {
        GO row, col;
        rowCol_from_linear_index(numGlobalElements_[0], linear_index, row, col);
        return elementIsOwned(row, col);
    }

  private:
    inline bool nodeIsOwned(GO row, GO col) const { return (myNodeStart_[0] <= row) && (row <= myNodeStop_[0]) && (myNodeStart_[1] <= col) && (col <= myNodeStop_[1]); }

    inline bool elementIsOwned(GO row, GO col) const { return (myElementStart_[0] <= row) && (row <= myElementStop_[0]) && (myElementStart_[1] <= col) && (col <= myElementStop_[1]); }

    inline GO linear_index_from_rowCol(GO total_num_x, GO row, GO col) const { return row * total_num_x + col; }

    inline void rowCol_from_linear_index(GO total_num_x, GO linear_id, GO &row, GO &col) const {
        // Convert linear index to [row,col] indices.
        // [in]  total_num_x: number of processes/elements/... per row.
        // [in]  linear_id: linear index
        // [out] row,colj: indices
        col = linear_id % total_num_x;               // column in which the linear index is
        row = (GO)((linear_id - col) / total_num_x); // row in which the linear index is
    }

    void initializeOwnedAndGhostNodeGlobalIDs(void);

    void initializeOwnedAndGhostElementGlobalIDs(void);

    // wrapped dual views
    global_ordinal_view_type ownedElementGlobalIDs_;
    global_ordinal_view_type ghostElementGlobalIDs_;

    global_ordinal_view_type ownedNodeGlobalIDs_;
    global_ordinal_view_type ghostNodeGlobalIDs_;

    global_ordinal_view_type ownedAndGhostNodeGlobalIDs_;
    global_ordinal_view_type ownedAndGhostElementGlobalIDs_;

    global_ordinal_2d_array_type ownedElementToNode_;
    global_ordinal_2d_array_type ghostElementToNode_;

    // Global Mesh Info
    LLA<GO> numGlobalElements_;
    LLA<GO> numGlobalNodes_;
    LLA<GO> numGlobalProcs_;

    // Local Mesh Info
    GO myElementStart_[2];
    GO myElementStop_[2];
    GO myNodeStart_[2];
    GO myNodeStop_[2];

    // Comm info
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    size_t myRank_;
    LLA<GO> rowColOfMyRank_;
};

template <typename GO, typename NO>
MeshDatabase<GO, NO>::MeshDatabase(Teuchos::RCP<const Teuchos::Comm<int>> comm, GO num_global_elements_x, GO num_global_elements_y, GO num_procs_x, GO num_procs_y)
    : numGlobalElements_(num_global_elements_y, num_global_elements_x), numGlobalNodes_(num_global_elements_y + 1, num_global_elements_x + 1), numGlobalProcs_(num_procs_y, num_procs_x), comm_(comm) {

    // Elements/nodes are numbered from left to right, from bottom to top of [0,1]^2.

    // Get processor decomp information
    myRank_ = comm_->getRank();
    rowCol_from_linear_index(numGlobalProcs_[1], myRank_, rowColOfMyRank_[0], rowColOfMyRank_[1]);

    // Get local element & node start / stop
    GO num_my_elements = 1, num_my_nodes = 1;
    for (int k = 0; k < 2; k++) {
        // Elements per rank. The last rank gets the remaining elements.
        // Example: 3 elements in x direction with 2 ranks --> rank 0: elements {0}, rank 1: elements {1,2}
        GO el_per_rank__k = numGlobalElements_[k] / numGlobalProcs_[k];

        // this rank: leftmost/rightmost column ID of elements
        // The different cases for the rightmost column ID are required, since the rightmost rank
        // may contain more elements than the others (remaining elements) if numGlobalElements_[k]
        // is not divisible by numGlobalProcs_[k].
        myElementStart_[k] = rowColOfMyRank_[k] * el_per_rank__k;
        myElementStop_[k] = (rowColOfMyRank_[k] == numGlobalProcs_[k] - 1) ? numGlobalElements_[k] - 1 : (rowColOfMyRank_[k] + 1) * el_per_rank__k - 1;
        num_my_elements *= (myElementStop_[k] - myElementStart_[k] + 1);

        // Nodes on the interface are assigend to the subdomain with the higher row/column index.
        myNodeStart_[k] = myElementStart_[k];
        myNodeStop_[k] = (rowColOfMyRank_[k] == numGlobalProcs_[k] - 1) ? myElementStop_[k] + 1 : myElementStop_[k];
        num_my_nodes *= (myNodeStop_[k] - myNodeStart_[k] + 1);
    }

    // Generate local-to-global map for element IDs.
    {
        ownedElementGlobalIDs_ = global_ordinal_view_type(globalDualViewType("ownedElementGlobalIDs_", num_my_elements));
        auto ownedElementGlobalIDs = ownedElementGlobalIDs_.getHostView(Tpetra::Access::ReadWrite);
        int local_linear_index = 0;
        for (GO row = myElementStart_[0]; row <= myElementStop_[0]; row++) {
            for (GO col = myElementStart_[1]; col <= myElementStop_[1]; col++) {
                GO global_linear_index = linear_index_from_rowCol(numGlobalElements_[1], row, col);
                ownedElementGlobalIDs(local_linear_index) = global_linear_index;
                local_linear_index++;
            }
        }
    }

    // Generate local-to-global map for node IDs.
    {
        ownedNodeGlobalIDs_ = global_ordinal_view_type(globalDualViewType("ownedNodeGlobalIDs_", num_my_nodes));
        auto ownedNodeGlobalIDs = ownedNodeGlobalIDs_.getHostView(Tpetra::Access::ReadWrite);
        int local_linear_index = 0;
        for (GO row = myNodeStart_[0]; row <= myNodeStop_[0]; row++) {
            for (GO col = myNodeStart_[1]; col <= myNodeStop_[1]; col++) {
                GO global_linear_index = linear_index_from_rowCol(numGlobalNodes_[1], row, col);
                ownedNodeGlobalIDs(local_linear_index) = global_linear_index;
                local_linear_index++;
            }
        }
    }

    // Generate the element list, i.e., elementList(row,:) = nodes-of-the-element.
    // Hardcoded for quads (4 nodes). Nodes are ordered counter-clockwise within an element.
    ownedElementToNode_ = global_ordinal_2d_array_type(global2DArrayDualViewType("ownedElementToNode_", num_my_elements));
    auto _ownedElementToNode = ownedElementToNode_.getHostView(Tpetra::Access::ReadWrite);
    int local_linear_index = 0;
    for (GO row = myElementStart_[0]; row <= myElementStop_[0]; row++) {
        for (GO col = myElementStart_[1]; col <= myElementStop_[1]; col++) {
            // The (i,j) of the bottom left corner matches for elements & nodes
            GO global_linear_index = linear_index_from_rowCol(numGlobalNodes_[1], row, col);

            // (d)---(c)
            //  |     |
            //  |     |
            // (a)---(b)
            _ownedElementToNode(local_linear_index, 0) = global_linear_index;                          // (a)
            _ownedElementToNode(local_linear_index, 1) = global_linear_index + 1;                      // (b)
            _ownedElementToNode(local_linear_index, 2) = global_linear_index + numGlobalNodes_[0] + 1; // (c)
            _ownedElementToNode(local_linear_index, 3) = global_linear_index + numGlobalNodes_[0];     // (d)

            local_linear_index++;
        }
    }

    // Generate the list of "ghost" elements.
    // Ghost elements are defined here as elements adjacent to the subdomain that contain
    // at least one node of the subdomain. Basically, this is the overlap of and overlapping
    // subdomain, where the overlap is one layer of elements.
    std::vector<GO> my_ghost_elements;
    for (GO row = myElementStart_[0] - 1; row <= myElementStop_[0] + 1; row++) {
        if (row < 0 || row >= numGlobalElements_[0])
            continue; // outside of mesh
        for (GO col = myElementStart_[1] - 1; col <= myElementStop_[1] + 1; col++) {
            if (col < 0 || col >= numGlobalElements_[1])
                continue; // outside of mesh

            // Ignore all elements of the subdomain.
            if ((row >= myElementStart_[0]) && (row <= myElementStop_[0]) && (col >= myElementStart_[1]) && (col <= myElementStop_[1]))
                continue;

            GO global_linear_index = linear_index_from_rowCol(numGlobalElements_[1], row, col);
            my_ghost_elements.push_back(global_linear_index);
        }
    }

    // Generate ghost element list, i.e., ghostElementList(row,:) = nodes-of-ghost-element.
    ghostElementGlobalIDs_ = global_ordinal_view_type(globalDualViewType("ghostElementGlobalIDs_", my_ghost_elements.size()));
    ghostElementToNode_ = global_ordinal_2d_array_type(global2DArrayDualViewType("ghostElementToNode_", my_ghost_elements.size()));
    auto _ghostElementGlobalIDs = ghostElementGlobalIDs_.getHostView(Tpetra::Access::ReadWrite);
    auto _ghostElementToNode = ghostElementToNode_.getHostView(Tpetra::Access::ReadWrite);
    for (size_t k = 0; k < my_ghost_elements.size(); k++) {
        GO col, row, elementID = my_ghost_elements[k];
        _ghostElementGlobalIDs(k) = elementID;
        rowCol_from_linear_index(numGlobalElements_[1], elementID, row, col);

        // (row, col) of the bottomleft node of an element matches (row, col) of the element.
        GO nodeID = linear_index_from_rowCol(numGlobalNodes_[1], row, col);

        // (d)---(c)
        //  |     |
        //  |     |
        // (a)---(b)
        _ghostElementToNode(k, 0) = nodeID;                          // (a)
        _ghostElementToNode(k, 1) = nodeID + 1;                      // (b)
        _ghostElementToNode(k, 2) = nodeID + numGlobalNodes_[1] + 1; // (c)
        _ghostElementToNode(k, 3) = nodeID + numGlobalNodes_[1];     // (d)
    }

    // Generate the list of "ghost" nodes, i.e., nodes that are part of the subdomain but that are not owned.
    // We use a set instead of a vector, since some nodes are added multiple times and we don't want duplicates.
    std::set<GO> my_ghost_nodes;
    auto ownedElementToNodeView = ownedElementToNode_.getHostView(Tpetra::Access::ReadOnly);
    for (size_t elID = 0; elID < ownedElementToNodeView.extent(0); elID++) {
        for (size_t col = 0; col < ownedElementToNodeView.extent(1); col++) {
            GO globalNodeID = ownedElementToNodeView(elID, col);
            if (!nodeIsOwned(globalNodeID)) {
                my_ghost_nodes.insert(globalNodeID);
            }
        }
    }

    // Convert set from above to vector.
    {
        ghostNodeGlobalIDs_ = global_ordinal_view_type(globalDualViewType("ghostNodeGlobalIDs_", my_ghost_nodes.size()));
        auto ghostNodeGlobalIDs = ghostNodeGlobalIDs_.getHostView(Tpetra::Access::ReadWrite);
        for (auto k = my_ghost_nodes.begin(); k != my_ghost_nodes.end(); k++) {
            size_t index_of_set_element = std::distance(my_ghost_nodes.begin(), k);
            ghostNodeGlobalIDs(index_of_set_element) = *k;
        }
    }

    initializeOwnedAndGhostNodeGlobalIDs();
    initializeOwnedAndGhostElementGlobalIDs();
}

template <typename GO, typename NO> void MeshDatabase<GO, NO>::initializeOwnedAndGhostNodeGlobalIDs(void) {
    size_t total_size = getNumOwnedNodes() + getNumGhostNodes();
    ownedAndGhostNodeGlobalIDs_ = global_ordinal_view_type(globalDualViewType("ownedAndGhostGlobalIDs_", total_size));
    auto _ownedAndGhostNodeGlobalIDs = ownedAndGhostNodeGlobalIDs_.getHostView(Tpetra::Access::ReadWrite);

    {
        size_t insert_idx = 0;
        auto ownedNodeGlobalIDs = getOwnedNodeGlobalIDs().getHostView(Tpetra::Access::ReadOnly);
        auto ghostNodeGlobalIDs = getGhostNodeGlobalIDs().getHostView(Tpetra::Access::ReadOnly);
        for (size_t idx = 0; idx < getNumOwnedNodes(); idx++) {
            _ownedAndGhostNodeGlobalIDs(insert_idx++) = ownedNodeGlobalIDs(idx);
        }
        for (size_t idx = 0; idx < getNumGhostNodes(); idx++) {
            _ownedAndGhostNodeGlobalIDs(insert_idx++) = ghostNodeGlobalIDs(idx);
        }
    }
}

template <typename GO, typename NO> void MeshDatabase<GO, NO>::initializeOwnedAndGhostElementGlobalIDs(void) {
    size_t total_size = getNumOwnedElements() + getNumGhostElements();
    ownedAndGhostElementGlobalIDs_ = global_ordinal_view_type(globalDualViewType("ownedAndGhostElementIDs_", total_size));
    auto _ownedAndGhostElementGlobalIDs = ownedAndGhostElementGlobalIDs_.getHostView(Tpetra::Access::ReadWrite);

    {
        size_t insert_idx = 0;
        auto ownedElementGlobalIDs = getOwnedElementGlobalIDs().getHostView(Tpetra::Access::ReadOnly);
        auto ghostElementGlobalIDs = getGhostElementGlobalIDs().getHostView(Tpetra::Access::ReadOnly);
        for (size_t idx = 0; idx < getNumOwnedElements(); idx++) {
            _ownedAndGhostElementGlobalIDs(insert_idx++) = ownedElementGlobalIDs(idx);
        }
        for (size_t idx = 0; idx < getNumGhostElements(); idx++) {
            _ownedAndGhostElementGlobalIDs(insert_idx++) = ghostElementGlobalIDs(idx);
        }
    }
}

template <typename GO, typename NO> void MeshDatabase<GO, NO>::print(std::ostream &outstream) {
    std::ostringstream ss, oss;
    ss << "[" << myRank_ << "," << rowColOfMyRank_[0] << "," << rowColOfMyRank_[1] << "]";
    oss << ss.str() << " Global Elements = [" << numGlobalElements_[0] << "x" << numGlobalElements_[1] << "] Nodes = [" << numGlobalNodes_[0] << "x" << numGlobalNodes_[1] << "]\n";
    oss << ss.str() << " Start/Stop Elements   = [" << myElementStart_[0] << "," << myElementStop_[0] << "]x[" << myElementStart_[1] << "," << myElementStop_[1] << "]\n";
    oss << ss.str() << " Start/Stop Nodes      = [" << myNodeStart_[0] << "," << myNodeStop_[0] << "]x[" << myNodeStart_[1] << "," << myNodeStop_[1] << "]\n";

    oss << ss.str() << " Owned Global Elements = ";
    {
        auto IDs = ownedElementGlobalIDs_.getHostView(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i < IDs.extent(0); i++) {
            oss << IDs[i] << " ";
        }
    }

    oss << "\n" << ss.str() << " Owned Global Nodes    = ";
    {
        auto IDs = ownedNodeGlobalIDs_.getHostView(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i < IDs.extent(0); i++) {
            oss << IDs[i] << " ";
        }
    }

    // Element list of size numElements x nodesPerElement
    oss << "\n" << ss.str() << " Owned Element2Node    = ";
    {
        auto IDs = ownedElementToNode_.getHostView(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i < IDs.extent(0); i++) {
            oss << "(";
            for (size_t j = 0; j < IDs.extent(1); j++) {
                oss << IDs(i, j) << " ";
            }
            oss << ") ";
        }
    }

    oss << "\n" << ss.str() << " Ghost Global Elements = ";
    {
        auto IDs = ghostElementGlobalIDs_.getHostView(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i < IDs.extent(0); i++) {
            oss << IDs[i] << " ";
        }
    }

    oss << "\n" << ss.str() << " Ghost Global Nodes    = ";
    {
        auto IDs = ghostNodeGlobalIDs_.getHostView(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i < IDs.extent(0); i++) {
            oss << IDs[i] << " ";
        }
    }

    oss << "\n" << ss.str() << " Ghost Element2Node    = ";
    {
        auto IDs = ghostElementToNode_.getHostView(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i < IDs.extent(0); i++) {
            oss << "(";
            for (size_t j = 0; j < IDs.extent(1); j++) {
                oss << IDs(i, j) << " ";
            }
            oss << ") ";
        }
    }

    oss << "\n" << ss.str() << " Owned And Ghost Nodes = ";
    {
        auto IDs = ownedAndGhostNodeGlobalIDs_.getHostView(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i < IDs.extent(0); i++) {
            oss << IDs[i] << " ";
        }
    }

    oss << "\n" << ss.str() << " Owned And Ghost Elements = ";
    {
        auto IDs = ownedAndGhostElementGlobalIDs_.getHostView(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i < IDs.extent(0); i++) {
            oss << IDs[i] << " ";
        }
    }

    outstream << oss.str() << std::endl;
}

template <typename GO, typename NO> void MeshDatabase<GO, NO>::exportToFiles(std::string base_file_name) const {

    // Get the matrix's communicator.
    Teuchos::RCP<const Teuchos::Comm<int>> comm = comm_;
    const int myRank = comm->getRank();

    std::ofstream out;
    std::string fname = "mesh_" + std::to_string(myRank) + ".txt";
    out.open(fname);

    //  const global_size_t numRows = rangeMap->getGlobalNumElements ();
    //  const global_size_t numCols = domainMap->getGlobalNumElements ();
    out << "process " << myRank << std::endl;
    out << "points" << std::endl;

    {
        auto IDs = ownedElementToNode_.getHostView(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i < IDs.extent(0); i++) {
            for (size_t j = 0; j < IDs.extent(1); j++) {
                out << IDs(i, j) << " ";
            }
            out << std::endl;
        }
    }
}

#endif // FROSCHEXAMPLES_ADAPTIVE_GDSW_MESHDATABASE_HPP
