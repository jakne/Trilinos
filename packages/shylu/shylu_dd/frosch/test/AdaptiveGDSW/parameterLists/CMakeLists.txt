TRIBITS_COPY_FILES_TO_BINARY_DIR(AdaptiveGDSW_CopyParameterListFolder
    DEST_FILES ParameterList_test1.xml ParameterList_test2.xml ParameterList_test3.xml
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
    EXEDEPS AdaptiveGDSW
)
