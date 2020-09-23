#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt to create the  unified class for importing dependecies by compiling previously developed functionality
Reason behind - to make something functional to search / import modules if the containing it directory tree isn't
included in any project

@author: ssklykov
"""
# %% Imports
import os
# import sys


# %% Class implementation
class projectImport():
    """Class for holding all support methods for importing / searching."""
    projectTree = {}
    anchorFolder = ""
    containingFolder = ""
    projectPath = ""
    osSpecificSeparator = ""

    def __init__(self, anchorFolder: str = ".git"):
        """
        Initialization along attempting to collect project structure.

        Parameters
        ----------
        anchorFolder : str, optional
            The anchor folder should be in the project main directory tree. The default is ".git" that is presented
            for version controlled projects in Git.
        Returns
        -------
        Sample of class, actually.

        """
        self.containingFolder = os.getcwd()
        self.osSpecificSeparator = os.sep
        iteration_depth = 5  # Iteration depth for operation of going in hierarchy of folders up
        iteration = 1
        # Searching and collecting absolute path to anchor folder - that on the same level of the whole project
        while (anchorFolder not in os.listdir(os.curdir)) and (iteration <= iteration_depth):
            os.chdir("..")
            iteration += 1
        if (iteration == 5) and (anchorFolder not in os.listdir(os.curdir)):
            print("Anchor folder didn't found")
            self.anchorFolder = None
            self.projectPath = self.containingFolder
        else:
            self.projectPath = os.path.abspath(os.curdir)
            self.anchorFolder = anchorFolder
            folderTree = os.listdir(self.projectPath)  # Get all files and folders in the expected project tree
            if len(folderTree) > 0:  # If project tree is non-empty
                folderTree = projectImport.deleteGitFolders(folderTree)
            else:
                self.projectTree["Project"] = "Empty"
            # Again, checking that it's not an empty git project
            if len(folderTree) > 0:
                self.projectTree["Project"] = folderTree
            else:
                self.projectTree["Project"] = "Empty"

    def includeFileToSearchPath(self, fileName: str) -> bool:
        """
        DEPRECATED!
        Make a walk along project tree and include specified file to import searching path.
        This feature is utilizing os.walk() functionality.
        Returns
        -------
        status:
            boolean operation result. False in case of file not found, class initialized in empty project, not such
            file found in a project structure. Otherwise - True.
        """
        if len(fileName) == 0:
            print("NoFileSpecified")
            return False
        folderTree = self.projectTree["Project"]
        if type(folderTree) == str:
            print("Class contains info about empty project")
            return False
        # if type(folderTree) == list:
        #     print("folder tree is a list type")
        if (type(folderTree) == list) and (len(folderTree)) > 0:
            for obj in folderTree:
                absPathObj = os.path.join(self.projectPath, obj)
                if os.path.isfile(absPathObj):
                    if obj == fileName:
                        projectImport.includeFileToSysPath(absPathObj)
                        return True
        return False

    def makeDirectoryGraph(self):
        # TODO: make a recursively call function for built directory graph
        # Assuming that initial list of files and folders in the project directory is non-empty
        for obj in self.folderTree["Project"]:
            # for all files and directories in the root project:
            absPathObj = os.path.join(self.projectPath, obj)  # get an absolute path
            # If it's a directory, then build another dictionary entry
            if os.path.isdir(absPathObj):
                (dirPath, dirList, fileList) = os.walk(absPathObj)

        pass

    def makeDictEntryForFolder(folderName: str, folderPath: str):
        pass

    # DEPRECATED METHOD (for reference)
    # @staticmethod
    # def isFileInDirectoryTree(fileName: str, absPathEntry: str) -> str:
    #     walkDone = False
    #     foundOnPath = ""
    #     while not walkDone:
    #         (dirPath, dirList, fileList) = os.walk(absPathEntry)
    #         if len(fileList) > 0:
    #             for file in fileList:
    #                 if fileName == file:
    #                     foundOnPath = dirPath
    #                     walkDone = True
    #                     break
    #         if (len(dirList) > 0) and (not walkDone):

    @staticmethod
    def includeFileToSysPath(absPathToFolder: str):
        import sys
        # print("sys.path before: ", sys.path)
        if (sys.path.count(absPathToFolder) == 0):
            sys.path.append(absPathToFolder)
        elif (sys.path.count(absPathToFolder) > 1):
            for i in (2, sys.path.count(absPathToFolder)):
                sys.path.remove(absPathToFolder)
        # print("sys.path after: ", sys.path)

    @staticmethod
    def deleteGitFolders(folderTree: list) -> list:
        """Deleting specific for git controlled projects folders .git and .gitignore from a project tree"""
        i = 0
        while i < len(folderTree):
            if (folderTree[i] == '.git') or (folderTree[i] == ".gitignore"):
                folderTree.pop(i)
                # print("Deleted folder", delF)
            else:
                i += 1
        return folderTree

    def printAllSelfValues(self):
        """
        For debugging / testing
        """
        print("The folder containing this script:", self.containingFolder)
        # print("specific folder separator:", self.osSpecificSeparator)
        print("Expected project tree containing the anchor:", self.projectPath)


# %% Testing some capabilities
if __name__ == "__main__":
    demo_instance = projectImport()
    demo_instance.printAllSelfValues()
    included = demo_instance.includeFileToSearchPath("LICENSE")
    # demo_instance2 = projectImport("Some non-exist folder")
    # demo_instance2.printAllSelfValues()
