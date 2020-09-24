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
    listOFAllPathsToDirs = []
    listOfFilesNames = []
    listOfDirs = []
    default_excluded_dirs = []

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
                self.projectTree[self.projectPath] = "Empty"
            # Again, checking that it's not an empty git project
            if len(folderTree) > 0:
                self.projectTree[self.projectPath] = folderTree
            else:
                self.projectTree[self.projectPath] = "Empty"
            if anchorFolder == ".git":
                self.default_excluded_dirs = [".git", ".gitignore"]

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
        return False

    def makeDirectoryGraph(self):
        # TODO: make a recursively call function for built directory graph
        # Assuming that initial list of files and folders in the project directory is non-empty
        for paths, dirs, files in os.walk(self.projectPath):
            self.listOfFilesNames.append(files)
            if (dirs != []):
                self.listOfDirs.append(dirs)

    def recursiveBuildDirGraph(self):
        # First step - checking project structure (ASSUMING that project is non-empty)
        # TODO: Now it isn't recursive!
        currentPath = self.projectPath
        rootPath = self.projectPath
        pathsCollection = []
        walkDone = False
        pathsCollection.append(currentPath)
        dirList = projectImport.getDirList(currentPath)
        firstIteration = True
        numDirsInRoot = 0
        jFolderInRoot = 0
        while (not walkDone):
            if len(dirList) > 0:
                parentPath = currentPath
                for folder in dirList:
                    path_to_append = os.path.join(parentPath, folder)
                    flag = True
                    if path_to_append not in pathsCollection:
                        pathsCollection.append(path_to_append)
                        # Exchanging of the currentPath only once below
                        if flag:
                            currentPath = path_to_append
                            flag = False
                        # Counting how many folders in the root directory
                        if firstIteration:
                            numDirsInRoot += 1
                firstIteration = False
                # print("new path:", currentPath)
                dirList = projectImport.getDirList(currentPath)
            else:
                if len(pathsCollection) > 1:
                    (parentPath, tail) = os.path.split(currentPath)
                    if (parentPath == rootPath):
                        pass

                else:
                    walkDone = True
                walkDone = True  # preventing infinite loop because the behaviour not implemented
                self.listOFAllPathsToDirs = pathsCollection

    def buildProjectDirGraph(self):
        # Second attempt to write recursive iterator building the project folder graph
        currentPath = self.projectPath
        rootPath = self.projectPath
        ignored_folders = self.default_excluded_dirs
        pathsCollection = []
        nodesCollection = []  # for accounting number and place of nodes visiting
        nodesVisited = []
        walkDone = False
        # First iteration not in the loop:
        numberOfDirs = len(projectImport.getDirList(rootPath))
        nodesVisited = [0 for i in range(numberOfDirs)]
        node = {}
        (root, tail) = os.path.split(rootPath)
        node[tail] = projectImport.getDirList(rootPath)
        nodesCollection.append(node)
        print(node)
        print(nodesVisited)
        # while (not walkDone):
        #     pass
        pass

    # For testing capabilities below is narrowing of output
    default_excluded_folders = [".git", ".gitignore", "Matricies_Manipulation", "Sorting", "Plotting",
                                "Experimental", "LinearSystems", "Interpolation", "FunctionRoots", "__pycashe__",
                                "MonteCarlo"]

    @staticmethod
    def getDirList(rootPath: str, excludingFolders: list = default_excluded_folders) -> list:
        """
        Returns of list of dirs down from specified path (specified path should point to some folder).

        Parameters
        ----------
        rootPath : str
            Path to the root directory
        excludingFolders : list, optional
            List with names of folders that must be excluded from returning list.
            The default is [".git, .gitignore"].

        Returns
        -------
        list
            List with folder names that the root directory contains.

        """
        if os.path.exists(rootPath):
            listOfDirs = os.listdir(rootPath)
            i = 0
            while i < len(listOfDirs):
                pathToObj = os.path.join(rootPath, listOfDirs[i])
                if (not os.path.isdir(pathToObj)) or (listOfDirs[i] in excludingFolders):
                    listOfDirs.pop(i)
                else:
                    i += 1
            return listOfDirs
        else:
            return []

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
        For debugging / testing - enabling it for get output
        """
        print("*********************************")
        # print("The folder containing this script:", self.containingFolder)
        # print("specific folder separator:", self.osSpecificSeparator)
        # print("Expected project tree containing the anchor:", self.projectPath)
        print("Collected paths to folders: ", self.listOFAllPathsToDirs)


# %% Testing some capabilities
if __name__ == "__main__":
    demo_instance = projectImport()
    # demo_instance.recursiveBuildDirGraph()
    # demo_instance.makeDirectoryGraph()
    # demo_instance.printAllSelfValues()
    demo_instance.buildProjectDirGraph()
    # included = demo_instance.includeFileToSearchPath("LICENSE")
    # demo_instance2 = projectImport("Some non-exist folder")
    # demo_instance2.printAllSelfValues()
