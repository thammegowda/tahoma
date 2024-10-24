
message(STATUS "Tests are enabled: ${BUILD_TESTING}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")


add_executable(tahoma-tests tahoma_tests.cpp ${BACKWARD_ENABLE})
target_link_libraries(tahoma-tests PRIVATE tahoma_lib)
add_backward(tahoma-tests)


set(TEST_CONFIG ${CMAKE_SOURCE_DIR}/examples/transformer-nmt.yaml)

add_test(NAME tahoma-tests:config_parse
    COMMAND tahoma-tests config_parse ${TEST_CONFIG}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_test(NAME tahoma-tests:read_lines
    COMMAND tahoma-tests read_lines ${TEST_CONFIG}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_test(NAME tahoma-tests:data_loader_sync
    COMMAND tahoma-tests data_loader_sync ${TEST_CONFIG}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_test(NAME tahoma-tests:data_loader_async
    COMMAND tahoma-tests data_loader_async ${TEST_CONFIG}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_test(NAME tahoma-tests:samples
    COMMAND tahoma-tests samples ${TEST_CONFIG}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})


