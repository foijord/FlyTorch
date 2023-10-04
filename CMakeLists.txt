cmake_minimum_required (VERSION 3.22)
project (FlyTorch VERSION 202311 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Boost REQUIRED)

set(TARGET FlyTorchTests)

add_compile_definitions(
	$<$<CONFIG:Debug>:DEBUG>
	$<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>									# no min/max macros in Windows.h
	$<$<CXX_COMPILER_ID:MSVC>:_WIN32_WINNT=0x0601>)						# target Windows >= 7 (Boost requires this)

add_compile_options(
	$<$<CXX_COMPILER_ID:MSVC>:/W3>										# warning level 4
	$<$<CXX_COMPILER_ID:MSVC>:/WX>)										# warnings as errors

add_executable(
	${TARGET}
	FlyTorch.h
	FlyTorchTests.h
	main.cpp)

target_include_directories(
	${TARGET} PRIVATE
	${Boost_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR})

target_link_directories(
	${TARGET} PRIVATE
	${Boost_LIBRARY_DIRS})

target_link_libraries(
	${TARGET} PRIVATE
	${Boost_LIBRARIES})
