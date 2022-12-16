#Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set(ADDON_NAME AgentIsolation)


find_library(${ADDON_NAME}_LIBRARY
  NAMES
    ${ADDON_NAME}
  PATHS
    ${PACKAGE_PREFIX_DIR}/addons/${ADDON_NAME}
  NO_DEFAULT_PATH
)
add_library(ASRCAISim1::${ADDON_NAME} SHARED IMPORTED)
set_target_properties(ASRCAISim1::${ADDON_NAME} PROPERTIES
    IMPORTED_LOCATION "${${ADDON_NAME}_LIBRARY}"
)
if(WIN32)
  set_target_properties(ASRCAISim1::${ADDON_NAME} PROPERTIES
    IMPORTED_IMPLIB "${${ADDON_NAME}_LIBRARY}"
  )
endif()
list(APPEND ASRCAISim1_LIBRARIES ASRCAISim1::${ADDON_NAME})

set_target_properties(ASRCAISim1 PROPERTIES
    INTERFACE_LINK_LIBRARIES ASRCAISim1::${ADDON_NAME}
)
