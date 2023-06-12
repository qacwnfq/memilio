# Install script for directory: /home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/_deps/spdlog-src

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/_deps/spdlog-src/include/" REGEX "/fmt\\/bundled$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so.1.11.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so.1.11"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib:/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/bin")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib/libspdlog.so.1.11.0"
    "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib/libspdlog.so.1.11"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so.1.11.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so.1.11"
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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so"
         RPATH "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib:/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/bin")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib/libspdlog.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so"
         OLD_RPATH ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
         NEW_RPATH "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/lib:/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/bin")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/spdlog/fmt/bundled/" TYPE DIRECTORY FILES "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/_deps/spdlog-src/include/spdlog/fmt/bundled/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/spdlog.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/spdlog/spdlogConfigTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/spdlog/spdlogConfigTargets.cmake"
         "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/_deps/spdlog-build/CMakeFiles/Export/5b93ed3f55f9d470264c31a33fb13826/spdlogConfigTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/spdlog/spdlogConfigTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/spdlog/spdlogConfigTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/spdlog" TYPE FILE FILES "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/_deps/spdlog-build/CMakeFiles/Export/5b93ed3f55f9d470264c31a33fb13826/spdlogConfigTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/spdlog" TYPE FILE FILES "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/_deps/spdlog-build/CMakeFiles/Export/5b93ed3f55f9d470264c31a33fb13826/spdlogConfigTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/spdlog" TYPE FILE FILES
    "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/_deps/spdlog-build/spdlogConfig.cmake"
    "/home/jadebeck/repos/LOKI/memilio/pycode/memilio-simulation/_skbuild/linux-x86_64-3.10/cmake-build/_deps/spdlog-build/spdlogConfigVersion.cmake"
    )
endif()

