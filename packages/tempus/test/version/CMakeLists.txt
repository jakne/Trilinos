SET(UNIT_TEST_DRIVER ${TEMPUS_UNIT_TEST_MAIN})

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  version
  SOURCES version.cpp ${UNIT_TEST_DRIVER}
  NUM_MPI_PROCS 1
  )

