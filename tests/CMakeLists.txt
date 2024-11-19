
message(STATUS "Tests are enabled: ${BUILD_TESTING}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")


add_executable(tahoma-tests tahoma_tests.cpp)
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
