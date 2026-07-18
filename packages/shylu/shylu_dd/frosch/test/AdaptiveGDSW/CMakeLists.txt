TRIBITS_ADD_EXECUTABLE(
    AdaptiveGDSW
    SOURCES Diffusion_Heterogeneous.cpp
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(AdaptiveGDSW_CopyFiles
    DEST_FILES ParameterList.xml
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
    EXEDEPS AdaptiveGDSW
)

ADD_SUBDIRECTORIES(
    parameterLists
)

TRIBITS_ADD_TEST(
    AdaptiveGDSW
    NAME test_AdaptiveGDSW_test1
    ARGS "--plist=parameterLists/ParameterList_test1.xml --test=1 --test_iter=10 --test_relResidual=5.603e-08 --test_condEst=4.883 --test_significantDigits=4"
    COMM mpi
    NUM_MPI_PROCS 4
)

TRIBITS_ADD_TEST(
    AdaptiveGDSW
    NAME test_AdaptiveGDSW_test2
    ARGS "--plist=parameterLists/ParameterList_test2.xml --test=1 --test_iter=10 --test_relResidual=8.920e-07 --test_condEst=4.925 --test_significantDigits=4"
    COMM mpi
    NUM_MPI_PROCS 4
)

TRIBITS_ADD_TEST(
    AdaptiveGDSW
    NAME test_AdaptiveGDSW_test3
    ARGS "--plist=parameterLists/ParameterList_test3.xml --test=1 --test_iter=32 --test_relResidual=5.333e-07 --test_condEst=23.77 --test_significantDigits=4"
    COMM mpi
    NUM_MPI_PROCS 100
)
