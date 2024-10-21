
#add_executable(TestInstantiator TestInstantiator.cxx)
#target_link_libraries(TestInstantiator vtkCommon)
#add_test(NAME TestInstantiator COMMAND TestInstantiator)

# add_test(
#     NAME "Test Train" 
#     COMMAND tahoma -c ${CMAKE_CURRENT_SOURCE_DIR}/examples/nmt/transformer.yaml )

#include_directories(
#    "${CMAKE_SOURCE_DIR}/src/include"
#)
# "${CMAKE_SOURCE_DIR}/src"

message(STATUS "Tests are enabled: ${BUILD_TESTING}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
    message(STATUS "    Include directory: ${dir}")
endforeach()


add_executable(test_data_loader test_data_loader.cpp)
# target_link_libraries(test_data_loader data_loader)

add_test(NAME test_data_loader
         COMMAND test_data_loader "arg1" "arg2")
