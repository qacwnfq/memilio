# Install script for directory: /home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/_deps/jsoncpp-src/src/lib_json

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so.1.9.5"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so.25"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib:/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/bin")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib/libjsoncpp.so.1.9.5"
    "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib/libjsoncpp.so.25"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so.1.9.5"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so.25"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
           NEW_RPATH "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib:/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/bin")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so"
         RPATH "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib:/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/bin")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib/libjsoncpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so"
         OLD_RPATH ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
         NEW_RPATH "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib:/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/bin")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjsoncpp.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib/libjsoncpp.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/objects-Release/jsoncpp_object" TYPE FILE FILES
    "json_reader.cpp.o"
    "json_value.cpp.o"
    "json_writer.cpp.o"
    FILES_FROM_DIR "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/_deps/jsoncpp-build/src/lib_json/CMakeFiles/jsoncpp_object.dir/./")
endif()

