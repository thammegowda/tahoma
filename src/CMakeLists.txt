#include_directories(.)
add_executable(rtgpp hello.cpp)
add_executable(mnist mnist.cpp)

target_link_libraries(rtgpp "${TORCH_LIBRARIES}")
target_link_libraries(mnist "${TORCH_LIBRARIES}")
#set_property(TARGET rtgpp PROPERTY CXX_STANDARD 20)


