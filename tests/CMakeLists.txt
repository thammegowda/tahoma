
#add_executable(TestInstantiator TestInstantiator.cxx)
#target_link_libraries(TestInstantiator vtkCommon)
#add_test(NAME TestInstantiator COMMAND TestInstantiator)

add_test(
    NAME "Test Train" 
    COMMAND rtgp -c ${CMAKE_CURRENT_SOURCE_DIR}/examples/nmt/transformer.yaml )