include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
    set(CCD_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
    set(CCD_EXTRA_OPTIONS "")
endif()

function(ccd_download_project name)
    download_project(
        PROJ         ${name}
        SOURCE_DIR   ${TIGHT_INCLUSION_EXTERNAL}/${name}
        DOWNLOAD_DIR ${TIGHT_INCLUSION_EXTERNAL}/.cache/${name}
        QUIET
        ${CCD_EXTRA_OPTIONS}
        ${ARGN}
    )
endfunction()

################################################################################


## Eigen
# function(ccd_download_eigen)
#     ccd_download_project(eigen
# 	GIT_REPOSITORY           https://gitlab.com/libeigen/eigen.git
# 	GIT_TAG       3.3.7
#     )
# endfunction()

# # Sampled CCD Queries
# function(ticcd_download_sample_queries)
#     ccd_download_project(Sample-Queries
#     GIT_REPOSITORY https://github.com/Continuous-Collision-Detection/Sample-Queries.git
#     GIT_TAG        4d6cce33477d8d5c666c31c8ea23e1aea97be371
#   )
# endfunction()