include(GoogleTest)

FetchContent_Declare(
  KaTestrophe
  GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
  GIT_TAG v1.0.2)

FetchContent_MakeAvailable(KaTestrophe)

add_executable(sorting_test sorting_test.cpp)
target_link_libraries(sorting_test PRIVATE dcx::dcx KaTestrophe::main)
katestrophe_add_mpi_test(sorting_test CORES 1 2 4)

add_executable(distribute_test distribute_test.cpp)
target_link_libraries(distribute_test PRIVATE dcx::dcx KaTestrophe::main)
katestrophe_add_mpi_test(distribute_test CORES 1 2 4)

add_executable(suffix_array_test suffix_array_test.cpp)
target_link_libraries(suffix_array_test PRIVATE dcx::dcx KaTestrophe::main)
katestrophe_add_mpi_test(suffix_array_test CORES 1 2 4)

add_executable(sa_check_test sa_check_test.cpp)
target_link_libraries(sa_check_test PRIVATE dcx::dcx KaTestrophe::main)
katestrophe_add_mpi_test(sa_check_test CORES 1 2 4)
