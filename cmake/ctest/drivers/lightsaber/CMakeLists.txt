TRILINOS_DRIVER_SETUP()

TRILINOS_DRIVER_ADD_DASHBOARD(
  SERIAL_RELEASE_DEFAULT
  ctest_linux_experimental_mpi_release_avatar_lightsaber.cmake
  CTEST_INSTALLER_TYPE release
  RUN_SERIAL
  TIMEOUT_MINUTES 330
  )

TRILINOS_DRIVER_ADD_DASHBOARD(
  SERIAL_RELEASE_FLOAT
  ctest_linux_experimental_mpi_release_float_lightsaber.cmake
  CTEST_INSTALLER_TYPE release
  RUN_SERIAL
  TIMEOUT_MINUTES 330


TRILINOS_DRIVER_ADD_DASHBOARD(
  SYCL_CPU_RELEASE
  ctest_linux_experimental_mpi_release_sycl_cpu_lightsaber.cmake
  CTEST_INSTALLER_TYPE release
  TIMEOUT_MINUTES 330
  )


TRILINOS_ADD_REQUIRED_CMAKE_INSTALLS()
