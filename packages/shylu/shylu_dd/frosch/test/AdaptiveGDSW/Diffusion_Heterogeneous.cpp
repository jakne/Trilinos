// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/*
 * Adaptive GDSW Test Problem
 *
 * Quickstart:
 * 1) nonadaptive
 * mpirun -np 9 --oversubscribe ./ShyLU_DDFROSch_Diffusion_Heterogeneous.exe --num-elements-1d=60 --coeff=1000000.0 --coeff_step=2 --nrows_leave_untouched=10 --overlap=0 --plist=ParameterList.xml --adaptive=0
 * 2) adaptive
 * mpirun -np 9 --oversubscribe ./ShyLU_DDFROSch_Diffusion_Heterogeneous.exe --num-elements-1d=60 --coeff=1000000.0 --coeff_step=2 --nrows_leave_untouched=10 --overlap=0 --plist=ParameterList.xml --adaptive=1
 *
 * Description:
 * 2D diffusion problem, finite-element discretization with bilinear basis functions, square domain,
 * zero Dirichlet boundary conditions at bottom and left, zero Neumann boundary condition elsewhere,
 * large coefficient of, e.g., 1000 in beams that intersect the domain but that do not touch the
 * the Dirichlet boundary, coefficient of 1 elsewhere.
 *
 * -div( rho(x,y)*gradient(u(x,y)) ) = 1  in [0,1]^2,
 *                                 u = 0  on D, D = {(x,y): x=0 v y=0}, 
 *                     gradient(u)*n = 0  on N, N = {(x,y): x=1 v y=1} \ {A,C} = boundary([0,1]^2) \ D,
 * where n is the outward unit normal vector field, N is the Neumann boundary, and D is the Dirichlet boundary.
 *
 *
 *                        Neumann boundary
 *                              ^
 *                              |
 *                        A___________B
 *                        |           |
 * Dirichlet boundary --> |           | --> Neumann boundary
 *                        |           |
 *                        |           |
 *                        D___________C
 *
 *                              |
 *                              v
 *                        Dirichlet boundary
 *
 * A = (0,1), B = (1,1), C = (1,0), D = (0,0)
 *
 * Coefficient function:
 * The base value is rho(x,y) = 1.
 * There are channels of high coefficients, e.g., rho(x,y) = 1e6.
 * The channels are in the columns of the structured quadrilateral mesh, for example every third column. They always start in the second column. There is never a high coefficient in the last column.
 * The first few rows (counting from the bottom, exact number can be specified) don't have a high coefficient.
 * Note: That the left column and bottom is left to rho(x,y) = 1 gives a more difficult problem. Otherwise, and specifically for small problems, the problems become much easier to solve.
 *
 * Example: |x| denotes a large coefficient.
 *
 *     ______________
 *    |  |x|  |x|    |
 *    |  |x|  |x|    |
 *    |  |x|  |x|    |
 *    |  |x|  |x|    |
 *    |  |x|  |x|    |
 *    |              |
 *    |______________|
 *
 */


// TODO: includes need to be cleaned up. It compiles, but which ones should be included here explicitly?

// System
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem>
// #include <mpi.h>
// #include <cmath>
// #include <iomanip>
// #include <sstream>

// Teuchos
#include <Teuchos_Array.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
// #include <Teuchos_FancyOStream.hpp>
// #include <Teuchos_TimeMonitor.hpp>
// #include <Teuchos_XMLParameterListCoreHelpers.hpp>

// Tpetra
#include <Tpetra_Assembly_Helpers.hpp>
#include <Tpetra_Details_DefaultTypes.hpp>
#include <Tpetra_FECrsMatrix_decl.hpp>
#include <Tpetra_FEMultiVector_decl.hpp>
#include <Tpetra_Map_decl.hpp>
#include <MatrixMarket_Tpetra.hpp>

// Thyra
#include <Thyra_LinearOpBase_decl.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_MultiVectorBase_decl.hpp>
#include <Thyra_SolveSupportTypes.hpp>

// Stratimikos
#include <Stratimikos_FROSch_def.hpp>

// Xpetra
#include <Xpetra_Map_decl.hpp>
#include <Xpetra_Matrix_decl.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
// #include <Xpetra_DefaultPlatform.hpp>
// #include <Xpetra_IO.hpp>
// #include <Xpetra_Parameters.hpp>

// FROSch
// #include <FROSch_Types.h>  // requires Xpetra namespace
#include <FROSch_Tools_def.hpp>

#include "fem_assembly_MeshDatabase.hpp"

using SC = Tpetra::Details::DefaultTypes::scalar_type;
using LO = Tpetra::Details::DefaultTypes::local_ordinal_type;
using GO = Tpetra::Details::DefaultTypes::global_ordinal_type;
using NO = Tpetra::Details::DefaultTypes::node_type;

using std::cout;
using std::endl;

// Element stiffness matrix for Q1 reference finite element.
//
// nodes of Q1 element:
//
//    3 ------ 2
//    |        |
//    |        |
//    0 ------ 1
//
// reference element stiffness matrix:
//
//        0   1   2   3
//       --------------
//    0 | 4  -1  -2  -1
//    1 |-1   4  -1  -2   divided by 6
//    2 |-2  -1   4  -1
//    3 |-1  -2  -1   4
//
template <typename ViewType, typename SC> KOKKOS_INLINE_FUNCTION void elementStiffnessMatOnRefElementQ1(ViewType &elementMatrix) {
    SC pattern[4][4] = {{4, -1, -2, -1}, {-1, 4, -1, -2}, {-2, -1, 4, -1}, {-1, -2, -1, 4}};
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            elementMatrix(i, j) = pattern[i][j] / 6.0;
        }
    }
}

// Element load vector for Q1 reference finite element.
template <typename SC> void elementLoadVecOnRefElementQ1(Teuchos::Array<SC> &rhs) {
    for (int i = 0; (int)i < rhs.size(); i++)
        rhs[i] = static_cast<SC>(0.25);
}

int main(int argc, char *argv[]) {
    Tpetra::ScopeGuard tpetraScope(&argc, &argv); // initializes MPI
    Teuchos::RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();

    // Default parameters
    int numElements1D = 8;  // number of finite elements along the x and y direction of the domain
    double coeff = 1.0e6;   // high coefficient, the remainder is set to 1
    GO coeff_step = 2;      // every nth column, starting with the second, a high coefficient is set
    GO nrows_leave_untouched = 4; // a coefficient of 1 is set in these bottom rows
    int overlap = 0;        // algebraic overlap of the domain decomposition method, 0 means only the interface nodes are shared
    int useAdaptiveCoarseSpace_int = 1; // use adaptive coarse space: 1 use, 0 don't use
    std::string xmlFile = "ParameterList.xml"; // default name of parameter list file

    // Read parameters from command line and from parameter list.
    Teuchos::CommandLineProcessor my_CLP;
    my_CLP.setOption("num-elements-1d", &numElements1D, "Number of elements to generate in the x and y directon of the 2D grid.");
    my_CLP.setOption("coeff", &coeff, "Coefficient value for heterogeneities; coeff >= 1.");
    my_CLP.setOption("coeff_step", &coeff_step, "Spacing between large coefficient beams.");
    my_CLP.setOption("nrows_leave_untouched", &nrows_leave_untouched, "Spacing between bottom Dirichlet boundary and beginning of large coefficient.");
    my_CLP.setOption("overlap", &overlap, "Overlap.");
    my_CLP.setOption("adaptive", &useAdaptiveCoarseSpace_int, "Use Adaptive Coarse Space (0: no, 1: yes).");
    my_CLP.setOption("plist", &xmlFile, "File name of the parameter list.");
    my_CLP.recogniseAllOptions(true);
    my_CLP.throwExceptions(false);
    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = my_CLP.parse(argc, argv);
    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED)
        return (EXIT_SUCCESS);
    bool useAdaptiveCoarseSpace = (useAdaptiveCoarseSpace_int != 0);

    // Check if file exists with name of parameter file.
    if (!std::filesystem::exists(xmlFile)) {
        if (comm->getRank() == 0) {
            std::cout << "Parameter file not found: " << xmlFile << std::endl;
        }
        return (EXIT_SUCCESS);
    }
    Teuchos::RCP<Teuchos::ParameterList> parameterList = Teuchos::getParametersFromXmlFile(xmlFile);

    comm->barrier();
    if (comm->getRank() == 0) {
        cout << "##################\n# Parameter List #\n##################" << endl;
        parameterList->print(cout);
        cout << endl;
    }
    comm->barrier();

    Teuchos::RCP<Teuchos::ParameterList> parameterList_linearSolver = Teuchos::sublist(parameterList, "Linear Solver");
#ifdef HAVE_FROSch_DEBUG
    Teuchos::RCP<Teuchos::ParameterList> parameterList_debug_main   = Teuchos::sublist(parameterList, "Debug Output: main");
#endif
    Teuchos::RCP<Teuchos::ParameterList> parameterList_main         = Teuchos::sublist(parameterList, "main");

    const int N = (int)(std::sqrt(comm->getSize()) + 100 * std::numeric_limits<double>::epsilon()); // number of subdomains in each coordinate direction
    if (comm->getSize() > N * N) {
        cout << "Number of ranks must be a square number, since the domain is decomposed into N x N subdomains." << endl;
        return EXIT_FAILURE;
    }

    comm->barrier();
    if (comm->getRank() == 0) {
        cout << "###################\n# Mesh #\n###################\n" << endl;
        cout << "#elements in x and y direction: " << numElements1D << "\n" << endl;
    }
    comm->barrier();

    // Generate structured mesh of rectangles for Q1 finite element discretization.       
    MeshDatabase<GO, NO> mesh(comm, numElements1D, numElements1D, N, N);
    if (parameterList_main->get("Export subdomain meshes to mesh_i.txt", false))
        mesh.exportToFiles("mesh_");

    comm->barrier();
    if (comm->getRank() == 0)
        cout << "###################\n# Assembly #\n###################\n" << endl;
    comm->barrier();

    // Build Tpetra Maps
    // https://trilinos.org/docs/dev/packages/tpetra/doc/html/classTpetra_1_1Map.html#a24490b938e94f8d4f31b6c0e4fc0ff77
    // The repeated map is given by the row map of the FECrsGraph.
    const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid();
    Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> row_map = Teuchos::rcp(new Tpetra::Map<LO, GO, NO>(INVALID, mesh.getOwnedNodeGlobalIDs().getDeviceView(Tpetra::Access::ReadOnly), 0, comm));
    auto domain_map = row_map;
    // auto range_map = row_map;
    Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> owned_plus_shared_map =
        Teuchos::rcp(new Tpetra::Map<LO, GO, NO>(INVALID, mesh.getOwnedAndGhostNodeGlobalIDs().getDeviceView(Tpetra::Access::ReadOnly), 0, comm));
#ifdef HAVE_FROSch_DEBUG
    if (parameterList_debug_main->get("Row Map for FE Graph", false))
        FROSch::debug::printMap(row_map, "Row Map for FE Graph", __FILE__, __LINE__);
    if (parameterList_debug_main->get("Owned-Plus-Shared Map for FE Graph", false))
        FROSch::debug::printMap(owned_plus_shared_map, "Owned-Plus-Shared Map for FE Graph", __FILE__, __LINE__);
#endif

    // Build FECrsGraph.
    Teuchos::RCP<Tpetra::FECrsGraph<LO, GO, NO>> fe_graph = Teuchos::rcp(new Tpetra::FECrsGraph<LO, GO, NO>(row_map, owned_plus_shared_map, 16));
    auto owned_element_to_node_gids = mesh.getOwnedElementToNode().getHostView(Tpetra::Access::ReadOnly);
    {
        Tpetra::beginAssembly(*fe_graph);  
        // for each finite element ...
        for (size_t element_gidx = 0; element_gidx < mesh.getNumOwnedElements(); element_gidx++) {
            // Define array for global node IDs of a finite element. Quadrilateral: 4 nodes.
            Teuchos::Array<GO> global_ids_in_row(4);
            // The following loop extracts the corresponding element list row from owned_element_to_node_gids.
            // global_ids_in_row = owned_element_to_node_gids(elementID,:);
            for (size_t element_node_idx = 0; element_node_idx < owned_element_to_node_gids.extent(1); element_node_idx++) {
                global_ids_in_row[element_node_idx] = owned_element_to_node_gids(element_gidx, element_node_idx);
            }
        
            // Add the contributions from the current row into the graph.
            // Example: If the current element contains the nodes [0,1,4,5], then for each of these nodes, we insert [0, 1, 4, 5].
            for (size_t element_node_idx = 0; element_node_idx < owned_element_to_node_gids.extent(1); element_node_idx++) {
                fe_graph->insertGlobalIndices(global_ids_in_row[element_node_idx], global_ids_in_row());
            }
        }
        Tpetra::endAssembly(*fe_graph); // fillComplete
    }

    // Assemble stiffness matrix and load vector.
    Teuchos::RCP<Tpetra::FECrsMatrix<SC, LO, GO, NO>> fe_matrix;
    Teuchos::RCP<Tpetra::FEMultiVector<SC, LO, GO, NO>> rhs;
    {
        // Note: hardcoded number 4: Q1 finite elements (quadrilaterals)

        const SC coefficient_max = coeff; // minimum coefficient is set to 1.0
        comm->barrier();
        if (comm->getRank() == 0) {
            cout << "Coefficient = " << coeff << "\n" << endl;
            cout << "Spacing between large coefficient beams = " << coeff_step << "\n" << endl;
        }
        comm->barrier();

        fe_matrix = Teuchos::rcp(new Tpetra::FECrsMatrix<SC, LO, GO, NO>(fe_graph));
        rhs = Teuchos::rcp(new Tpetra::FEMultiVector<SC, LO, GO, NO>(domain_map, fe_graph->getImporter(), 1));

        Kokkos::View<SC[4][4], MeshDatabase<GO, NO>::hostType> element_matrix("element_matrix"); // 4 x 4 element stiffness matrix
        Teuchos::Array<SC> element_rhs(4); // 4 x 1 element load vector

        // local-to-global map of element indices (not node indices)
        auto l2g_elements = mesh.getOwnedElementGlobalIDs().getHostView(Tpetra::Access::ReadWrite);

        const int nodesPerElem = mesh.getOwnedElementToNode().extent(1);
        const GO nex = numElements1D;  // [n]umber of [e]lements in [x] direction
        // const GO ney = nex;
        SC h = 1.0 / (SC)nex; // finite element diameter (unit square with nex elements in x/y direction)

        // Loop over elements.
        Tpetra::beginAssembly(*fe_matrix, *rhs);
        for (std::size_t element_idx = 0; element_idx < mesh.getNumOwnedElements(); ++element_idx) {

            // Compute the contributions for the current element (element stiffness matrix and element load vector).
            elementStiffnessMatOnRefElementQ1<Kokkos::View<SC[4][4], MeshDatabase<GO, NO>::hostType>, SC>(element_matrix);
            elementLoadVecOnRefElementQ1<SC>(element_rhs);
            // Up to this point, we have the contributions of the reference element.
            // Next, we scale them to obtain the contributions of the physical elements.
            // The element matrix is the same (in this case) for the reference and the physical element.
            for (int ind = 0; ind < nodesPerElem; ind++) {
                element_rhs[ind] *= h * h;
            }
            
            // list of global column ids of a row
            // global_column_ids = owned_element_to_node_gids(elementID,:);
            Teuchos::Array<GO> global_column_ids(4);
            for (int element_node_idx = 0; element_node_idx < nodesPerElem; ++element_node_idx) {
                global_column_ids[element_node_idx] = owned_element_to_node_gids(element_idx, element_node_idx);
            }

            // For each node of the current element:
            // * add value to load vector,
            // * fill a values array with the corresponding values from the element stiffness matrix,
            // * add the values array to the global stiffness matrix.
            for (int element_node_idx = 0; element_node_idx < 4; ++element_node_idx) {
                const GO global_row_id = owned_element_to_node_gids(element_idx, element_node_idx);

                // load vector
                if ((global_row_id >= nex + 1) && (global_row_id % (nex + 1) != 0)) {
                    rhs->sumIntoGlobalValue(global_row_id, 0, element_rhs[element_node_idx]);
                    // else: Zero Dirichlet boundary at the bottom and left of [0,1]^2.
                }

                // stiffness matrix
                Teuchos::Array<SC> scalar_values_of_row(4); // scalar values for each column in a row
                for (int col_idx = 0; col_idx < 4; col_idx++) {
                    // if ( (global_column_ids[col_idx] == global_row_id) && ( (global_row_id < 1) )) {
                    if ((global_column_ids[col_idx] == global_row_id) && ((global_row_id < nex + 1) || (global_row_id % (nex + 1) == 0))) {
                        // diagonal entry corresponding to Dirichlet boundary node: add 1.0
                        scalar_values_of_row[col_idx] = 1.0;
                    //} else if ( (global_row_id < 1) || (global_column_ids[col_idx] < 1) ) {
                    } else if ((global_row_id < nex + 1) || (global_row_id % (nex + 1) == 0) || (global_column_ids[col_idx] < nex + 1) || (global_column_ids[col_idx] % (nex + 1) == 0)) {
                        // off-diagonal entry corresponding to Dirichlet boundary node: set to 0.0
                        scalar_values_of_row[col_idx] = 0.0;
                    } else {
                        scalar_values_of_row[col_idx] = element_matrix(element_node_idx, col_idx);
                    }

                    // Introduce some heterogeneity by modifying the bilinear form a(u,v) = integral dot( grad(u), grad(v) ) d(x,y)
                    // to a(u,v) = integral coefficient(x,y) * dot( grad(u), grad(v) ) d(x,y).
                    // The coefficient function describes beams of high coefficients that intersect the entire domain.
                    // The beams do not touch the Dirichlet boundary (this would make the problem easier); we leave the first nrows_leave_untouched rows untouched.
                    // Every coeff_step element column will have a coefficient of coefficient_max; the remainder is set to 1.
                    // The first and last element columns are also set to 1.
                    // Example: global columns 0,1,2,...,7 --> {1,3,5} are set to coefficient_max
                    //          global columns 0,1,2,...,6 --> {1,3,5} are set to coefficient_max
                    const GO gid_el = l2g_elements(element_idx); // global element index
                    const GO id_in_row = gid_el % nex;
                    const bool isInLastColumn = (id_in_row % (nex - 1) == 0);
                    if (!isInLastColumn && (id_in_row % coeff_step == 1) && (gid_el >= nrows_leave_untouched * nex)) {
                        scalar_values_of_row[col_idx] *= coefficient_max;
                    }
                }

                fe_matrix->sumIntoGlobalValues(global_row_id, global_column_ids, scalar_values_of_row);
            }
        }
    }

    // Up to this point, we have not called fillComplete. Thus, the matrices on the ranks are Neumann stiffness matrices.
    // Here, we take a snapshot of the current state of the FECrsMatrix such that we later have access to the local Neumann matrices.
    Teuchos::RCP<Tpetra::FECrsMatrix<SC, LO, GO, NO>> neumann_matrices = FROSch::FECrsMatrix_deepCopy(fe_matrix.getConst(), fe_graph);

#ifdef HAVE_FROSch_DEBUG
    if (parameterList_debug_main->get("Row Map of Stiffness Matrix before endAssembly", false))
        FROSch::debug::printMap(fe_matrix->getRowMap(), "Row Map of Stiffness Matrix before endAssembly", __FILE__, __LINE__);
    if (parameterList_debug_main->get("Col Map of Stiffness Matrix before endAssembly", false))
        FROSch::debug::printMap(fe_matrix->getColMap(), "Col Map of Stiffness Matrix before endAssembly", __FILE__, __LINE__);
    if (parameterList_debug_main->get("Local Neumann Matrices", false))
        FROSch::debug::printTpetraCrsMatrix_local_sparse(fe_matrix, "Local Neumann Matrices (fe_matrix before endAssembly)", __FILE__, __LINE__);
#endif

    {
        Tpetra::endAssembly(*fe_matrix); 
        Tpetra::endAssembly(*rhs);
    }

#ifdef HAVE_FROSch_DEBUG
    if (parameterList_debug_main->get("Row Map Stiffness Matrix", false))
        FROSch::debug::printMap(fe_matrix->getRowMap(), "Row Map Stiffness Matrix", __FILE__, __LINE__);
    if (parameterList_debug_main->get("Repeated Map", false))
        FROSch::debug::printMap(neumann_matrices->getRowMap(), "Repeated Map = Row Map of Local Neumann Matrices", __FILE__, __LINE__);
#endif

    // Convert Tpetra::FEMultiVector to Xpetra::MultiVector.
    Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO>> rhs_tpetra = Teuchos::rcp_dynamic_cast<Tpetra::MultiVector<SC, LO, GO, NO>>(rhs);
    Teuchos::RCP<Xpetra::TpetraMultiVector<SC, LO, GO, NO>> rhs_xpetra_tpetra = Teuchos::rcp(new Xpetra::TpetraMultiVector<SC, LO, GO, NO>(rhs_tpetra));
    Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> rhs_xpetra = Teuchos::rcp_dynamic_cast<Xpetra::MultiVector<SC, LO, GO, NO>>(rhs_xpetra_tpetra);

    if (parameterList_main->get("Export load vector to fem_rhs.txt", false)) {
        // Unlike a matrix, the multivector is written "unmapped"; that is, it is sorted by local IDs, rank by rank, and not by global IDs.
        // The map (map_fem_rhs.txt) needs to be used to map the values in post. It is exported to map_fem_rhs.txt.
        Xpetra::IO<SC, LO, GO, NO>::Write("fem_rhs.txt", *rhs_xpetra);
    }

    // Convert Tpetra::FECrsMatrix to Xpetra::Matrix.
    Teuchos::RCP<Tpetra::CrsMatrix<SC, LO, GO, NO>> crs_tpetra = Teuchos::rcp_dynamic_cast<Tpetra::CrsMatrix<SC, LO, GO, NO>>(fe_matrix);
    Teuchos::RCP<Xpetra::TpetraCrsMatrix<SC, LO, GO, NO>> crs_xpetra_tpetra = Teuchos::rcp(new Xpetra::TpetraCrsMatrix<SC, LO, GO, NO>(crs_tpetra));
    Teuchos::RCP<Xpetra::CrsMatrix<SC, LO, GO, NO>> crs_xpetra = Teuchos::rcp_dynamic_cast<Xpetra::CrsMatrix<SC, LO, GO, NO>>(crs_xpetra_tpetra);
    Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> matrix_xpetra = Teuchos::rcp(new Xpetra::CrsMatrixWrap(crs_xpetra));

    if (parameterList_main->get("Export stiffness matrix to fem_matrix.txt", false)) {
        std::ofstream ofs("fem_matrix.txt", std::ofstream::out);
        Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<SC, LO, GO, NO>>::writeSparse(ofs, fe_matrix);
        // Xpetra::IO<SC, LO, GO, NO>::Write("fem_matrix.txt", *matrix_xpetra, true);
    }

#ifdef HAVE_FROSch_DEBUG
    if (parameterList_debug_main->get("Stiffness Matrix", false))
        FROSch::debug::printXpetraMatrix_local_dense(matrix_xpetra, "Stiffness Matrix", __FILE__, __LINE__);
#endif

    comm->barrier();
    if (comm->getRank() == 0)
        cout << "##############################\n# Assembly Monolithic System #\n##############################\n" << endl;
    comm->barrier();

    Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> K = matrix_xpetra;

    // Initialize solution vector with zero.
    Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> solution_xpetra = Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(K->getDomainMap(), 1); // one column multivector
    solution_xpetra->putScalar(Teuchos::ScalarTraits<SC>::zero()); // This will store the solution and is used for the initial vector of the iterative solver.

    // Convert stiffness matrix, load vector, and solution vector to Thyra objects.
    Xpetra::CrsMatrixWrap<SC, LO, GO, NO>& crsWrapK = dynamic_cast<Xpetra::CrsMatrixWrap<SC, LO, GO, NO>&>(*K);
    Teuchos::RCP<const Thyra::LinearOpBase<SC>> K_thyra = Xpetra::ThyraUtils<SC, LO, GO, NO>::toThyra(crsWrapK.getCrsMatrix());
    Teuchos::RCP<Thyra::MultiVectorBase<SC>> solution_thyra = Teuchos::rcp_const_cast<Thyra::MultiVectorBase<SC>>(Xpetra::ThyraUtils<SC, LO, GO, NO>::toThyraMultiVector(solution_xpetra));
    Teuchos::RCP<const Thyra::MultiVectorBase<SC>> rhs_thyra = Xpetra::ThyraUtils<SC, LO, GO, NO>::toThyraMultiVector(rhs_xpetra);

    // Pass info to FROSch via parameter list.
    Teuchos::RCP<Teuchos::ParameterList> plList = sublist(parameterList_linearSolver, "Preconditioner Types");
    sublist(plList, "FROSch")->set("Dimension", 2);
    sublist(plList, "FROSch")->set("Overlap", overlap);
    sublist(plList, "FROSch")->set("Repeated Map", neumann_matrices->getRowMap());
    sublist(plList, "FROSch")->set("DofOrdering", "NodeWise"); // 2D vector problem: (x1,y1,x2,y2,...,xn,yn) instead of (x1,x2,...,xn,y1,y2,...,yn).
    sublist(plList, "FROSch")->set("DofsPerNode", 1);
    sublist(sublist(plList, "FROSch"), "GDSWCoarseOperator")->set("Neumann Matrices", neumann_matrices);
    sublist(sublist(plList, "FROSch"), "GDSWCoarseOperator")->set("Use Adaptive Coarse Space", useAdaptiveCoarseSpace);

    comm->barrier();
    if (comm->getRank() == 0) {
        cout << "##################\n# Parameter List Linear Solver #\n##################" << endl;
        parameterList_linearSolver->print(cout);
        cout << endl;
    }
    comm->barrier();

    comm->barrier();
    if (comm->getRank() == 0)
        cout << "###################################\n# Stratimikos LinearSolverBuilder #\n###################################\n" << endl;
    comm->barrier();

    Stratimikos::LinearSolverBuilder<SC> linearSolverBuilder;
    Stratimikos::enableFROSch<LO, GO, NO>(linearSolverBuilder);
    linearSolverBuilder.setParameterList(parameterList_linearSolver);

    comm->barrier();
    if (comm->getRank() == 0)
        cout << "######################\n# Thyra PrepForSolve #\n######################\n" << endl;
    comm->barrier();

    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<SC>> lowsFactory = linearSolverBuilder.createLinearSolveStrategy("");
    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
    lowsFactory->setOStream(out);
    lowsFactory->setVerbLevel(Teuchos::VERB_HIGH);

    comm->barrier();
    if (comm->getRank() == 0)
        cout << "###########################\n# Thyra LinearOpWithSolve #\n###########################" << endl;
    comm->barrier();

    Teuchos::RCP<Thyra::LinearOpWithSolveBase<SC>> lows = linearOpWithSolve(*lowsFactory, K_thyra);

    comm->barrier();
    if (comm->getRank() == 0)
        cout << "\n#########\n# Solve #\n#########" << endl;
    comm->barrier();

    Thyra::SolveStatus<SC> status = Thyra::solve<SC>(*lows, Thyra::NOTRANS, *rhs_thyra, solution_thyra.ptr());

    if (parameterList_main->get("Export solution to fem_sol.txt", false)) {
        Xpetra::IO<SC, LO, GO, NO>::Write("fem_sol.txt", *solution_xpetra); // also writes map_fem_sol.txt
    }

    comm->barrier();
    if (comm->getRank() == 0)
        cout << "\n#############\n# Finished! #\n#############" << endl;
    comm->barrier();

    return EXIT_SUCCESS;
}
