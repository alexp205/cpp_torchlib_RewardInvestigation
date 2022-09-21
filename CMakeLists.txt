cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dqn)

find_package(Torch REQUIRED)

add_executable(dqn dqn.cpp)
target_link_libraries(dqn "${TORCH_LIBRARIES}")
set_property(TARGET dqn PROPERTY CXX_STANDARD 14)

if (MSVC)
	file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	add_custom_command(TARGET dqn
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${TORCH_DLLS}
		$<TARGET_FILE_DIR:dqn>)
endif(MSVC)
