
message(STATUS "Tests are enabled: ${BUILD_TESTING}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")


add_executable(tahoma-tests main.cpp)
target_link_libraries(tahoma-tests PUBLIC tahoma_lib Backward::Interface)
#add_backward(tahoma-tests)


set(TEST_CONFIG ${CMAKE_SOURCE_DIR}/examples/transformer-nmt.yaml)

# tests that require a config file to run
set(CONFIG_BASED_TESTS
    config_parse
    read_lines
    read_examples
    data_loader_sync
    data_loader_async
    samples
)

include(CTest)
foreach(test ${CONFIG_BASED_TESTS})
    add_test(NAME ${test}
        COMMAND tahoma-tests ${test} ${TEST_CONFIG}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()


add_test(NAME npz_load
    COMMAND tahoma-tests npz_load ${CMAKE_CURRENT_SOURCE_DIR}/data/test.npz
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# TODO: add tests for model persistence and loading

# pytest based tests
add_test(NAME pytests
    COMMAND python -m pytest -vs ${CMAKE_SOURCE_DIR}/tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_property(TEST pytests PROPERTY LABELS "pytests")
set_property(TEST pytests PROPERTY ENVIRONMENT "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")