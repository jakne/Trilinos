TRIBITS_ADD_EXECUTABLE(
    AdaptiveGDSW
    SOURCES agdsw.cpp
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
    ARGS "--plist=parameterLists/ParameterList_test1.xml"
    COMM mpi
    NUM_MPI_PROCS 4
)

TRIBITS_ADD_TEST(
    AdaptiveGDSW
    NAME test_AdaptiveGDSW_test2
    ARGS "--plist=parameterLists/ParameterList_test2.xml"
    COMM mpi
    NUM_MPI_PROCS 4
)

TRIBITS_ADD_TEST(
    AdaptiveGDSW
    NAME test_AdaptiveGDSW_test3
    ARGS "--plist=parameterLists/ParameterList_test3.xml"
    COMM mpi
    NUM_MPI_PROCS 100
)
